from math import sqrt

def manhatten_distance(rating1, rating2):
    """
    z = |x1-y1| + |x2-y2| + .... +|xn+yn|
    for case that data is dense
    """
    distance = 0
    for key in rating1:
        # only calculate the similarity based on rating of same thing.
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])

    return distance


def minkowski(rating1, rating2, r):
    """
    z = (|x1-y1|^r+.....+|xn + yn|^r)^(1/r)
    for case that data is dense
    """
    distance = 0
    common_ratings = False

    for key in rating1:
        if key in rating2:
            distance += pow(abs(rating1[key]-rating2[key]), r)
            common_ratings = True
    if common_ratings:
        return pow(distance, 1/r)
    else:
        return 0

def pearson_correlation(rating1, rating2):
    """
    for case is subject to grade-inflation

    """
    n = 0
    sum_multiple = 0
    avg_multiple_sum = 0
    sum1 = 0
    sum2 = 0

    pow_sum1 = 0
    pow_sum2 = 0

    for key in rating1:
        if key in rating2:
            n += 1
            value1 = rating1[key]
            value2 = rating2[key]

            pow_sum1 += pow(value1, 2)
            pow_sum2 += pow(value2, 2)

            sum1 += value1
            sum2 += value2
            sum_multiple += value1*value2

    avg_multiple_sum = (sum1 * sum2)/n

    denominator = sqrt((pow_sum1 - pow(sum1,2)/n)*(pow_sum2 - pow(sum2, 2)/n))

    if denominator == 0:
        return 0

    result = (sum_multiple - avg_multiple_sum) / denominator

    return result


def consine(rating1, rating2):
    """
    for case which data is sparse
    """
    pow_value1 = 0
    pow_value2 = 0

    multiple_value = 0

    for key in rating1:
        value1 = rating1[key]
        value2  = rating2[key] if key in rating2 else 0

        pow_value1 += pow(value1, 2)
        pow_value2 += pow(value2, 2)
        multiple_value += value1 * value2


    denominator = sqrt(pow_value1) * sqrt(pow_value2)
    if denominator == 0:
        return 0
    return multiple_value / denominator
