import torch


def get_random_rectangular_masks(
    b,
    c=1,
    w=224,
    h=224,
    min_size_w=40,
    min_size_h=40,
    max_size_w=80,
    max_size_h=80
):
    masks = []
    for i in range(b*c):

        x1 = torch.randint(0, w-min_size_w, (1, ))
        y1 = torch.randint(0, h-min_size_h, (1, ))
        x2 = x1 + torch.randint(min_size_w,
                                min(w-x1.item(), max_size_w), (1, ))
        y2 = y1 + torch.randint(min_size_h,
                                min(h-y1.item(), max_size_h), (1, ))

        mask = torch.zeros(w, h)
        mask[x1:x2, y1:y2] = 1

        masks.append(mask)
    masks = torch.stack(masks)
    masks = masks.reshape(b, c, w, h)
    return masks
