likes = {}
weight = 0.05
def store_rating(index, rate):
    like_count = like.get(index, 0)
    likes[index] = like_count + 1 if rate else like_count - 1

def order_with_like(regression):
    for i in range(len(regression)):
        index = regression[i][1]
        index_like = likes[index]
        if index_like is not None:
            regression[i][0] = regression[i][0] + weight * index_like

    return sorted(regression)
