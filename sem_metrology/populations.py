from __future__ import annotations

import numpy as np


def infer_feature_classes(values: np.ndarray, max_classes: int = 3,
                          requested_classes: int | None = None) -> dict:
    """Conservatively propose repeat-feature populations.

    The result is intentionally empty unless a multi-class model has adequate
    population, BIC improvement, separation, and membership in every class.
    Inputs may contain one or several physically meaningful size dimensions.
    """
    raw=np.asarray(values,dtype=float)
    if raw.ndim==1: raw=raw[:,None]
    finite=np.all(np.isfinite(raw),axis=1); clean=raw[finite]
    n,d=clean.shape
    empty={"class_count":1,"labels":None,"confidence":0.0,"reason":"single population retained"}
    if n<8: return {**empty,"reason":"N<8; class inference suppressed"}
    median=np.median(clean,axis=0); scale=np.median(np.abs(clean-median),axis=0)*1.4826
    fallback=np.std(clean,axis=0,ddof=1)
    scale=np.where(scale>1e-12,scale,np.where(fallback>1e-12,fallback,1.0))
    data=(clean-median)/scale

    def fit(k: int):
        order=np.argsort(data[:,0])
        seeds=[data[order[min(n-1,round((i+.5)*n/k-.5))]].copy() for i in range(k)]
        centers=np.asarray(seeds)
        labels=np.zeros(n,dtype=int)
        for _ in range(60):
            distances=((data[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            new_labels=np.argmin(distances,axis=1)
            if np.array_equal(labels,new_labels) and _>0: break
            labels=new_labels
            for index in range(k):
                members=data[labels==index]
                if len(members): centers[index]=members.mean(axis=0)
        residual=data-centers[labels]; rss=float(np.sum(residual**2))+1e-12
        bic=n*d*np.log(rss/(n*d))+k*(d+1)*np.log(n)
        return labels,centers,rss,bic

    baseline=fit(1); best=None
    candidates=([requested_classes] if requested_classes and requested_classes>1
                else range(2,min(max_classes,n//3)+1))
    for k in candidates:
        if k>n//3: continue
        labels,centers,rss,bic=fit(k)
        counts=np.bincount(labels,minlength=k)
        if counts.min()<max(3,round(.08*n)): continue
        residual=data-centers[labels]
        within=np.sqrt(np.mean(residual**2,axis=0))
        within=np.where(within>1e-6,within,1e-6)
        normalized_centers=centers/within
        pairwise=np.sqrt(((normalized_centers[:,None,:]-normalized_centers[None,:,:])**2).sum(axis=2))
        pairwise=pairwise[np.triu_indices(k,1)]
        separation=float(pairwise.min()) if len(pairwise) else 0.0
        improvement=float(baseline[3]-bic)
        # Automatic splitting is intentionally strict: adjacent class centroids
        # must be at least five pooled within-class RMS widths apart. This keeps
        # ordinary broad/normal process variation as one population.
        minimum_separation=2.00 if requested_classes else 5.00
        minimum_improvement=0.0 if requested_classes else 8.0
        if separation<minimum_separation or improvement<minimum_improvement: continue
        candidate=(bic,k,labels,centers,counts,separation,improvement)
        if best is None or candidate[0]<best[0]: best=candidate
    if best is None: return empty
    _,k,labels,centers,counts,separation,improvement=best
    # Stable class IDs run from the smallest to largest primary dimension.
    rank=np.argsort(centers[:,0]); remap=np.empty(k,dtype=int); remap[rank]=np.arange(k)
    labels=remap[labels]; centers=centers[rank]; counts=counts[rank]
    full_labels=np.full(len(raw),-1,dtype=int); full_labels[np.flatnonzero(finite)]=labels
    confidence=float(np.clip(.45+.07*(separation-5.0)+.012*(improvement-8),.5,.98))
    return {"class_count":k,"labels":full_labels,"confidence":confidence,
            "centers":centers*scale+median,"counts":counts,
            "separation_sigma":separation,"bic_improvement":improvement,
            "reason":f"{k} statistically separated size populations proposed"}
