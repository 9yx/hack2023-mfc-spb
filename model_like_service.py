likes = {}
weight = 0.015


def store_rating(index, rate):
    like_count = likes.get(index, 0)
    likes[index] = like_count + 1 if rate else like_count - 1


def distance_with_likes(distance):
    # Учет like и dislike
    for i in range(len(distance)):
        index = distance[i][1]
        index_like = likes.get(index)
        if index_like is not None:
            distance[i] = (distance[i][0] - weight * index_like, distance[i][1])

    return sorted(distance)
