def sum_of_attributions(attrs, indices):
    attrs = attrs.flatten(1)
    mask_attrs = attrs.gather(dim=1, index=indices)
    return mask_attrs.sum(dim=1, keepdim=True)