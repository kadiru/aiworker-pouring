"""Refine a neural cup proposal against the known Material A outline and band."""
import numpy as np
import torch
from torch.nn.functional import max_pool2d

from bottle_pour import CUP, SIDES, WALL
from perception_followup import VerifiedCupTracker, resize_coarse


# The visual marker uses the nominal cup profile and fixed dimensions in
# vessel_appearance.py, including when the glass collision shell is resized.
BAND_Z = .030
BAND_RADIUS = float(np.interp(BAND_Z, [z for z, _ in CUP], [r for _, r in CUP])) + .00252
ANGLES = np.arange(SIDES) * 2*np.pi/SIDES
VERTICES = np.stack((np.cos(ANGLES), np.sin(ANGLES)), axis=1)
FRACTION = np.arange(4)[:, None]/4
POLYGON = (VERTICES[:, None, :]*(1-FRACTION) + np.roll(VERTICES, -1, axis=0)[:, None, :]*FRACTION).reshape(-1,2)
DX, DY = np.meshgrid(np.arange(-20,21), np.arange(-20,21))
OFFSETS = np.stack((DX.ravel(),DY.ravel()),axis=1)
DISTANCE = np.linalg.norm(OFFSETS, axis=1)


def project_points(points, transform, intrinsics):
    view = (points-transform[:3,3]) @ transform[:3,:3]
    return view[..., :2] * [intrinsics[0,0],-intrinsics[1,1]] / -view[...,2,None] + intrinsics[:2,2]


def ring(centers, height, radius, transform, intrinsics):
    xyz = np.broadcast_to(height, (len(centers), len(POLYGON), 1))
    points = np.concatenate((centers[:,None,:]+radius*POLYGON, xyz), axis=2)
    uv = project_points(points, transform, intrinsics)
    center = project_points(np.c_[centers, np.full(len(centers),height)], transform, intrinsics)
    tangent = np.roll(uv,-1,axis=1)-np.roll(uv,1,axis=1)
    normal = tangent[...,::-1]*[1.,-1.]
    if np.mean(np.sum(normal*(uv-center[:,None,:]), axis=-1)) < 0.:
        normal = -normal
    normal /= np.linalg.norm(normal,axis=-1)[...,None]
    return uv, normal, uv[...,1] < center[:,None,1]+1.


def bilinear(image, uv):
    limits = np.array(image.shape[::-1]) - 1.001
    clipped = np.clip(uv, 0, limits)
    ij = clipped.astype(int)
    x, y = np.moveaxis(ij, -1, 0)
    dx, dy = np.moveaxis(clipped-ij, -1, 0)
    return ((1-dx)*((1-dy)*image[y,x]+dy*image[y+1,x])
            + dx*((1-dy)*image[y,x+1]+dy*image[y+1,x+1]))


def refine_outline(image, proposal, calibration, geometry):
    """Find the projected rim center near a camera-derived proposal."""
    transform = np.asarray(calibration["T"])
    intrinsics = np.asarray(calibration["K"])
    z = calibration["table_z"]
    uv = proposal+OFFSETS
    local = (uv-intrinsics[:2,2]) / [intrinsics[0,0],-intrinsics[1,1]]
    rays = np.c_[local,-np.ones(len(local))] @ transform[:3,:3].T
    depth = (z+geometry["rim_z_m"]-transform[2,3])/rays[:,2]
    centers = transform[:2,3] + rays[:,:2]*depth[:,None]
    rgb = image.astype(np.float32)
    gray = rgb @ np.array([.299,.587,.114],dtype=np.float32)
    blue = rgb[:,:,2]-rgb[:,:,0]

    def edge(height, radius, rear):
        p, n, back = ring(centers,z+height,radius,transform,intrinsics)
        mask = back if rear else ~back
        contrast = (bilinear(gray,p+1.5*n)-bilinear(gray,p-1.5*n)).clip(-15.,35.)
        return (contrast*mask).sum(1)/mask.sum(1)

    rim = edge(geometry["rim_z_m"], geometry["rim_radius_m"]+WALL, rear=True)
    base = edge(.5*geometry["bottom_z_m"], geometry["profile"][0][1]+WALL, rear=False)
    p, n, _ = ring(centers,z+BAND_Z,BAND_RADIUS,transform,intrinsics)
    chroma = (bilinear(blue,p)-.5*(bilinear(blue,p+4.*n)+bilinear(blue,p-4.*n))).clip(-10.,30.).mean(1)
    score = rim + .5*chroma + .2*base - .10*DISTANCE
    best = int(score.argmax())
    separated = np.linalg.norm(OFFSETS-OFFSETS[best],axis=1) >= 5.
    margin = float(score[best]-score[separated].max())
    if rim[best] < 12. or margin < 2. or np.max(np.abs(OFFSETS[best])) == 20:
        return None
    return uv[best]


class OutlineCupTracker(VerifiedCupTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.startup_frames = 0

    def shifted_startup_score(self, image, point):
        # Strided heatmaps can give a correct rim a weak peak at one pixel alignment.
        # Both shifted views must pass presence and agree in original image coordinates.
        threshold = self.artifact["threshold"]
        radius = self.artifact["consensus"]["radius_px"]
        if self.presence_score(image, point) < .90:
            return 0.
        width = image.shape[1]
        padded = np.pad(image, ((0,0),(4,4),(0,0)), mode="edge")
        scores = []
        for offset in (-4, 4):
            view = padded[:,4-offset:4-offset+width]
            detected, score = super().detect(view, refine=True)
            if not (score >= threshold and np.linalg.norm(detected - [offset,0] - point) <= radius):
                return 0.
            scores.append(score)
        return min(scores)

    @torch.inference_mode()
    def alternate_startup(self, image):
        # A distractor can outrank the real cup. Seek separated heatmap peaks
        # that agree across alignments before checking the photographed rim.
        threshold = self.artifact["threshold"]
        settings = self.artifact["consensus"]
        shifts = [(dx,dy) for dy in (0,-4,4) for dx in (0,-4,4)]
        h, w = image.shape[:2]
        padded = np.pad(image, ((4,4),(4,4),(0,0)), mode="edge")
        views = [resize_coarse(padded[4-dy:4-dy+h, 4-dx:4-dx+w]) for dx,dy in shifts]
        output = self.coarse(torch.as_tensor(np.stack(views), device=self.device))
        heat = output[:, :1].sigmoid()
        peaks = heat.masked_fill(heat != max_pool2d(heat, 7, 1, 3), 0)
        scores, locations = peaks.flatten(1).topk(3)
        if float(scores.max()) < threshold:
            return None
        y, x = locations // heat.shape[-1], locations % heat.shape[-1]
        batch = torch.arange(len(views), device=self.device)[:,None]
        subpixel = output.permute(0,2,3,1)[batch,y,x,1:].sigmoid()
        pixels = (torch.stack((x,y),dim=-1) + subpixel) * (w / heat.shape[-1])
        points = pixels.cpu().numpy() - np.asarray(shifts)[:,None,:]
        scores = scores.cpu().numpy()
        for view, peak in np.argwhere(scores >= threshold):
            point = points[view,peak]
            support = (scores >= settings["score_floor"]) & (np.linalg.norm(points-point,axis=2) <= settings["radius_px"])
            support[view] = False
            if not support.any() or self.presence_score(image,point) < .90:
                continue
            center = np.median(np.vstack((point,points[support])),axis=0)
            rim = refine_outline(image,center,self.artifact["calibration"],self.geometry)
            if rim is None or np.linalg.norm(rim-center) > 6.:
                continue
            presence = self.presence_score(image,rim)
            if presence >= .90:
                return rim, min(float(scores[view,peak]),presence)
        return None

    def detect(self, image, refine=True):
        point, score = super().detect(image, refine=refine)
        if refine and self.anchor is None:
            self.startup_frames += 1
            if (self.startup_frames > 12 and
                    self.artifact["consensus"]["score_floor"] <= score < self.artifact["threshold"]):
                score = max(score, self.shifted_startup_score(image, point))
            if self.startup_frames > 12 and score < self.artifact["threshold"]:
                alternate = self.alternate_startup(image)
                if alternate is not None:
                    return alternate
        if refine and score >= self.artifact["threshold"]:
            candidate = refine_outline(image, point, self.artifact["calibration"], self.geometry)
            if candidate is not None:
                # The decorative band can produce a second strong peak below
                # the rim. Reject large jumps when the two learned scales
                # already agree on the center.
                limit = self.artifact.get("outline_jump_guard",
                                          {"max_outline_jump_px": 6., "max_neural_gap_px": 2.5})
                if np.linalg.norm(candidate-point) > limit["max_outline_jump_px"]:
                    coarse, confidence = super().detect(image, refine=False)
                    if (confidence >= self.artifact["threshold"]
                            and np.linalg.norm(coarse-point) <= limit["max_neural_gap_px"]):
                        return point, score
                point, score = candidate, min(score,self.presence_score(image,candidate))
        return point, score
