from data import users
from recommdation import manhatten_distance, minkowski,\
pearson_correlation, consine


def computeNearestNeighbor(username, users):
    """
    creates a sorted list of users based on their distance to username
    """
    distances = []
    for user in users:
        if user != username:
            distance = minkowski(users[user], users[username], 2)
            distances.append((distance, user))

    distances.sort()
    return distances


def recommend(username, users):
    """
    recommendation applicable scenairo is:
    1. find similarity based on ratings from user and other users on same stuff
    2. get similar person
    3. find stuff similar person rated but user hasn't rated
    those stuff will be recommended. Namely, recommendation is the process that
    find similar person and recommend stuff that those similar person like
    or interested in to user.
    """
    # first find nearest neighbor
    nearest = computeNearestNeighbor(username, users)[0][1]
    recommendations = []

    # find bands neighbor rated that user didn't
    neighborRatings = users[nearest]
    userRatings = users[username]
    for artist in neighborRatings:
        # if user didn't rate, then store as recommedation
        if not artist in userRatings:
            recommendations.append((artist, neighborRatings[artist]))

    return sorted(recommendations, key=lambda artistTup: artistTup[1], reverse=True)


def cal_correlation(username, user):
    result = pearson_correlation(users[username], users[user])
    return result


def main():
    """
    main function
    """
    # distances = computeNearestNeighbor('Hailey', users)
    # print distances

    # recommendation = recommend('Hailey', users)
    # print recommendation

    # correlation = cal_correlation('Angelica', 'Bill')
    # print correlation
    # correlation = cal_correlation('Angelica', 'Hailey')
    # print correlation

    consin_value = consine(users['Angelica'], users['Veronica'])
    print consin_value


if __name__ == '__main__':
    main()
