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