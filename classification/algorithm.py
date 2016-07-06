"""
algorithm module
"""


def manhattan(vector_o, vector_d):
    """
    computes the manhattan distance,  z = |x1-y1| + |x2-y2| + .... +|xn+yn|
    :@param vector_o : the first vector (attrs list)
    :@param vector_d : the second vector (attrs list)
    """
    distance = 0
    for index in xrange(len(vector_o)):
        distance += abs(vector_o[index] - vector_d[index])
    
    return distance
    

def compute_nearest_neighbor(item_name, item_vector, items):
    """
    create a sorted list of items based on their distance to item
    :@param item_name: item name
    :@param item_vector: item vector
    :@param items: vector list
    """
    distances = []
    for item in items:
        if item != item_name:
            distance = manhattan(item_vector, items[item])
            distances.append((distance, item))
    
    distances.sort()

    return distances


def get_median(a_list):
    """
    get median of a list
    """
    if not a_list:
        return None
    
    blist = sorted(a_list)
    length = len(a_list)
    # when length is odd return middle element
    if length % 2 == 1:
        return blist[int(((length+1)/2)-1)]
    else: # when length is even, return average of the 2 middle elements
        v1 = blist[int(length/2)]
        v2 = blist[int((length/2)-1)]

        return (v1 + v2) / 2.0


def get_abs_standard_deviation(a_list, the_median):
    """
    get absolute standard deviation
    :@param alist: the data list
    :@param median: the median 
    """
    sum_value = 0
    for item in a_list:
        sum_value += abs(item - the_median)
    
    return sum_value / len(a_list)


test_users ={"Angelica": {"Dr Dog/Fate": "L", "Phoenix/Lisztomania": "L", 
            "Heartless Bastards/Out at Sea": "D", 
            "Todd Snider/Don't Tempt Me": "D", 
            "The Black Keys/Magic Potion": "D", 
            "Glee Cast/Jessie's Girl": "L", 
            "La Roux/Bulletproof": "D", 
            "Mike Posner": "D", 
            "Black Eyed Peas/Rock That Body": "D", 
            "Lady Gaga/Alejandro": "L"}, 
    "Bill": {
        "Dr Dog/Fate": "L", "Phoenix/Lisztomania": "L", 
        "Heartless Bastards/Out at Sea": "L", 
        "Todd Snider/Don't Tempt Me": "D", 
        "The Black Keys/Magic Potion": "L", 
        "Glee Cast/Jessie's Girl": "D", 
        "La Roux/Bulletproof": "D", "Mike Posner": "D", 
        "Black Eyed Peas/Rock That Body": "D", 
        "Lady Gaga/Alejandro": "D"} }

test_items ={"Dr Dog/Fate": [2.5, 4, 3.5, 3, 5, 4, 1],
        "Phoenix/Lisztomania": [2, 5, 5, 3, 2, 1, 1],
        "Heartless Bastards/Out at Sea": [1, 5, 4, 2, 4, 1, 1],
        "Todd Snider/Don't Tempt Me": [4, 5, 4, 4, 1, 5, 1],
        "The Black Keys/Magic Potion": [1, 4, 5, 3.5, 5, 1, 1],
        "Glee Cast/Jessie's Girl": [1, 5, 3.5, 3, 4, 5, 1],
        "La Roux/Bulletproof": [5, 5, 4, 2, 1, 1, 1],
        "Mike Posner": [2.5, 4, 4, 1, 1, 1, 1],
        "Black Eyed Peas/Rock That Body": [2, 5, 5, 1, 2, 2, 4],
        "Lady Gaga/Alejandro": [1, 5, 3, 2, 1, 2, 1]}

def classify_test(user, item_name, item_vector):
    """
    """
    nearest = compute_nearest_neighbor(item_name, item_vector, test_items)[0][1]
    rating = test_users[user][nearest]

    return rating


if __name__ == '__main__':
    print classify_test('Angelica', 'Cagle', [1, 5, 2.5, 1, 1, 5, 1])
    alist = [54, 72, 78, 49, 65, 63, 75, 67, 54]
    median = get_median(alist)
    print median
