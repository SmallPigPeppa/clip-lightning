import numpy



def i2t(images, captions, caps_per_image=5):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    import pdb;pdb.set_trace()
    npts = images.shape[0] / caps_per_image

    index_list = []
    npts = int(npts)

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        im = images[caps_per_image * index].reshape(1, images.shape[1])
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        rank = 1e20
        for i in range(caps_per_image * index, caps_per_image * index + caps_per_image, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1


def t2i(images, captions, caps_per_image=5):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    # if npts is None:
    #     caps_per_image = 5
    # else:
    #     caps_per_image = 2

    npts = images.shape[0] / caps_per_image

    ims = numpy.array([images[i] for i in range(0, len(images), caps_per_image)])
    npts = int(npts)

    ranks = numpy.zeros(caps_per_image * npts)
    top1 = numpy.zeros(caps_per_image * npts)
    for index in range(npts):

        queries = captions[caps_per_image * index:caps_per_image * index + caps_per_image]


        d = numpy.dot(queries, ims.T)
            
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[caps_per_image * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[caps_per_image * index + i] = inds[i][0]

    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    return r1