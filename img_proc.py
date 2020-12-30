def poly_to_mask(json_file, shape):
    mask = np.zeros(shape, dtype=np.int32)
    for json_obj in json_file:
        vertices = []
        vertices += [vertex for vertex in json_obj['geometry']['coordinates'][0]]
        vertices = np.array([vertices], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 1)
    return mask.astype(np.uint8) * 255


def resize_img(img, mul_resize=0.5):
    print(f'origin : {img.shape}')
    if img.shape[-1] == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape

    img = cv2.resize(img, (int(h * mul_resize), int(w * mul_resize)))
    print(f'resized : {img.shape}')
    return img


def crop_img(img, size=256, dup_ratio=0.2):
    dup_size = round(size * dup_ratio)
    jump_size = round(size * (1 - dup_ratio))

    if img.shape[-1] == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape

    # padding
    pad_h = jump_size - ((h - dup_size) % jump_size)
    pad_w = jump_size - ((w - dup_size) % jump_size)

    print(f'pad_h : {pad_h} / pad_w : {pad_w}')

    if img.shape[-1] == 3:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    else:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), 'constant')

    print(f'padded shape : {img.shape}')

    cropped = []
    i = 0
    for s_v in range(dup_size, h + pad_h, jump_size):
        for s_h in range(dup_size, w + pad_w, jump_size):
            cropped.append(img[s_v - dup_size:s_v + jump_size, s_h - dup_size:s_h + jump_size])

    return cropped, pad_h, pad_w, dup_size, jump_size


def poly_to_mask(json_file, shape):
    mask = np.zeros(shape, dtype=np.int32)
    for json_obj in json_file:
        vertices = []
        vertices += [vertex for vertex in json_obj['geometry']['coordinates'][0]]
        vertices = np.array([vertices], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 1)
    return mask.astype(np.uint8) * 255


def resize_img(img, mul_resize=0.5):
    print(f'origin : {img.shape}')
    if img.shape[-1] == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape

    img = cv2.resize(img, (int(h * mul_resize), int(w * mul_resize)))
    print(f'resized : {img.shape}')
    return img


def crop_img(img, size=256, dup_ratio=0.2):
    dup_size = round(size * dup_ratio)
    jump_size = round(size * (1 - dup_ratio))

    if img.shape[-1] == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape

    # padding
    pad_h = jump_size - ((h - dup_size) % jump_size)
    pad_w = jump_size - ((w - dup_size) % jump_size)

    print(f'pad_h : {pad_h} / pad_w : {pad_w}')

    if img.shape[-1] == 3:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    else:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), 'constant')

    print(f'padded shape : {img.shape}')

    cropped = []
    i = 0
    for s_v in range(dup_size, h + pad_h, jump_size):
        for s_h in range(dup_size, w + pad_w, jump_size):
            cropped.append(img[s_v - dup_size:s_v + jump_size, s_h - dup_size:s_h + jump_size])

    return cropped, pad_h, pad_w, dup_size, jump_size


def uncrop_img(masks, dest_h, dest_w, pad_h, pad_w, dup_size, jump_size, scale_ratio):
    size = masks[0].shape[0]

    if masks[0].shape[-1] == 3:
        pred_mask = resize_img(np.zeros((dest_h, dest_w, 3)), SCALE_RATIO)
        pred_mask = np.pad(pred_mask, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        h, w, c = pred_mask.shape
    else:
        pred_mask = resize_img(np.zeros((dest_h, dest_w)), SCALE_RATIO)
        pred_mask = np.pad(pred_mask, ((0, pad_h), (0, pad_w)), 'constant')
        h, w = pred_mask.shape

    i = 0
    for s_v in range(dup_size, h, jump_size):
        for s_h in range(dup_size, w, jump_size):
            pred_mask[s_v - dup_size:s_v + jump_size, s_h - dup_size:s_h + jump_size] += masks[i]
            i += 1

    for s_v in range(size, h, jump_size):
        pred_mask[s_v - dup_size:s_v, :] /= 2
    for s_h in range(size, w, jump_size):
        pred_mask[:, s_h - dup_size:s_h] /= 2

    return pred_mask[:-pad_h, :-pad_w]


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    '''
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
