"""Train a local ridge readout for the settled surface near the rim from aligned cup RGB crops."""
from cache_wide_mixed_r27b_surface import DEST, read, save, sha
import numpy as np
import torch
from torch.nn import functional as F

TRAIN = (20,22,24,26,27)


def surface_features(crops, cell):
    # Keep the liquid and rim in normalized cup coordinates. Discard the hand and table.
    rgb = crops[:,12:60,16:80].permute(0,3,1,2).float()/255.
    red,green,blue = rgb.unbind(1)
    planes = torch.stack((red-green,red-blue,(red+green+blue)/3),dim=1)
    return F.avg_pool2d(planes,cell,cell).flatten(1)


def fit(x,y,weight,alpha):
    w = weight/weight.sum()
    center = w@x
    ybar = w@y
    centered = x-center
    gram = centered.T@(w[:,None]*centered)
    gram.diagonal().add_(alpha)
    beta = torch.linalg.solve(gram,centered.T@(w*(y-ybar)))
    return beta,ybar-center@beta


def main():
    assert torch.cuda.device_count()==1 and "4070" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    rows=read(DEST/"settled-crops.json")
    with np.load(DEST/"settled-crops.npz") as z:
        images=torch.from_numpy(z["images"]).cuda()
        y=torch.tensor(z["heights"],device="cuda",dtype=torch.float64)
    rev=np.array([int(r["name"].split("_r")[1].split("_")[0]) for r in rows])
    groups={}
    for i,r in enumerate(rows):
        groups.setdefault((r["name"],r["pause"]),[]).append(i)
    pair=np.array(["_09_swap" in r["name"] for r in rows])
    train=np.isin(rev,TRAIN) & pair
    devs = sorted({int(i) for i in rev[train]})
    # Weight each distinct observed level plateau once per episode.
    weights=np.ones(len(y))
    for (_,pause),ids in groups.items():
        weights[ids]=1/(len(ids)*(9 if rev[ids[0]]==27 and pause>=5 else 1))
    w=torch.as_tensor(weights,device="cuda",dtype=y.dtype)
    candidates=[]
    cache={}
    for cell in (4,8,16):
        x=surface_features(images,cell).double()
        cache[cell]=x
        for alpha in (1e-6,1e-5,1e-4,.001,.01):
            errs=[]
            for v in devs:
                m=train & (rev!=v)
                b,k=fit(x[m],y[m],w[m],alpha)
                held=train & (rev==v) & (y.cpu().numpy()>=.85)
                err=(x[held]@b+k-y[held]).abs()
                errs.append(float(err.mean()))
            candidates.append(dict(cell=cell,alpha=alpha,cv_mean_mae=float(np.mean(errs)),cv_max_mae=max(errs)))
    chosen=min(candidates,key=lambda c:c["cv_mean_mae"]+c["cv_max_mae"])
    cell,alpha=chosen["cell"],chosen["alpha"];x=cache[cell]
    b,k=fit(x[train],y[train],w[train],alpha)
    pred=(x@b+k).cpu().numpy()
    checks=[]
    for (name,n),ids in groups.items():
        if not pair[ids[0]]:continue
        # Controller uses the lower quartile over its latest six valid settled views.
        ids=ids[-6:]
        true=float(y[ids].median())
        low=float(np.quantile(pred[ids],.25))
        checks.append(dict(name=name,pause=n,split="train" if train[ids[0]] else "validation",
                           predicted_low=low,true_median=true))
    # A positive reserve covers systematic overread on calibration episodes.
    overs=[c["predicted_low"]-c["true_median"] for c in checks
           if c["split"]=="train" and .86<=c["true_median"]<.91]
    reserve=max(.002,max(overs)+.001)
    bad=[c for c in checks if (c["predicted_low"]>=.90+reserve and c["true_median"]<.90)
         or (c["predicted_low"]<.90+reserve and c["true_median"]>=.908)]
    terminal={c["name"]:c for c in checks}
    passed=(not bad and all(c["predicted_low"]>=.9+reserve for n,c in terminal.items()
                    if c["true_median"]>=.908) and any(c["true_median"]<.90 for c in terminal.values()))
    print(chosen,"reserve",reserve,"bad",bad[:5],flush=True)
    artifact=DEST/"surface-readout.pt"
    torch.save(dict(cell=cell,weights=b.float().cpu(),intercept=float(k),minimum=.90+reserve),artifact)
    save(DEST/"surface-fit.json",dict(passed=passed,chosen=chosen,candidates=candidates,checks=checks,bad=bad,
         scope="Settled cup-detail RGB, pair 09 only. Grouped episode holdout.",reserve=reserve,
         train_windows=sum(c["split"]=="train" for c in checks),validation_windows=sum(c["split"]=="validation" for c in checks),
         training_frames=int(train.sum()),validation_frames=int((~train & pair).sum()),artifact_sha256=sha(artifact),
         frames_sha256=sha(DEST/"settled-crops.npz"),metadata_sha256=sha(DEST/"settled-crops.json")))
    assert passed


if __name__=="__main__":
    main()
