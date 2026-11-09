"""Independent convex-polyhedron separating-axis check, CPU geometry only."""
import numpy as np
from scipy.spatial import ConvexHull


def axes_and_edges(points):
    hull=ConvexHull(points)
    pairs={tuple(sorted((int(a),int(b)))) for face in hull.simplices
           for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0]))}
    edges=np.asarray([points[b]-points[a] for a,b in pairs])
    edges=edges[np.linalg.norm(edges,axis=1)>1e-12]
    return hull.equations[:,:3],edges


def convex_overlap(a,b,tolerance=1e-8):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite hull')
    origin=(a.mean(0)+b.mean(0))/2
    a=a-origin;b=b-origin
    na,ea=axes_and_edges(a);nb,eb=axes_and_edges(b)
    cross=np.cross(ea[:,None,:],eb[None,:,:]).reshape(-1,3)
    lengths=np.linalg.norm(cross,axis=1);cross=cross[lengths>1e-12]/lengths[lengths>1e-12,None]
    axes=np.concatenate((na,nb,cross))
    separation=-float('inf')
    for start in range(0,len(axes),256):
        axis=axes[start:start+256];pa=a@axis.T;pb=b@axis.T
        gaps=np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0))
        separation=max(separation,float(gaps.max()))
    return dict(intersect=separation<=tolerance,largest_separation_m=separation,
                axes_tested=len(axes),tolerance_m=tolerance)
