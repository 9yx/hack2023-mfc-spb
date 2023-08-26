likes = {}
weight = 0.015
def store_rating(index, rate):
    like_count = likes.get(index, 0)
    likes[index] = like_count + 1 if rate else like_count - 1

def order_with_like(regression):
    for i in range(len(regression)):
        index = regression[i][1]
        index_like = likes.get(index)
        if index_like is not None:
            regression[i] =(regression[i][0] - weight * index_like, regression[i][1])

    return sorted(regression)
