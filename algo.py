from typing import List, Tuple

def max_pair_durations(durationsMin: List[int], flightDurationMin: int) -> tuple[int, int]:
    
    T = flightDurationMin - 30
    n = len(durationsMin)
    
    if n < 2:
        return (-1, -1)
    
    #keep origianl indices, osrt by duration
    with_idx = sorted((t, i) for i, t in enumerate(durationsMin))
    best_i, best_j = -1, -1
    best_sum = -1
    best_longest = -1
    
    l, r = 0, n-1
    while l < r:
        t_l, i_l = with_idx[l]
        t_r, i_r = with_idx[r]
        s = t_l + t_r 
        if s > T:
            r -= 1
            continue
        
        #s<=T is valid; check/update best
        longest=max(t_l, t_r)
        if s > best_sum or (s == best_sum and longest > best_longest):
            best_sum = s
            best_longest = longest
            best_i, best_j = i_l, i_r
            
        l +=1
        
    return (best_i, best_j)

def test_max_pair_durations():
    durations = [90, 85, 75, 60, 120, 150, 125]
    d = 250
    res = max_pair_durations(
        durations,
        d
    )
    
    assert set(res) == {0, 6}, f"expected (0, 6) in any order, got{res}"
    print("Test passed: best pair is", res)    
    

#test 1st question    
test_max_pair_durations()


def is_valid(st) -> bool:
    m = {")": "(", "}": "{", "]": "["}
    openSet = set(m.values())
    buf = []
    
    for c in st:
        if c in openSet:
            buf.append(c)
        elif c in m:
            if buf == [] or buf.pop()!=m[c]:
                return False
        else:
            return False
    return not buf        

def test_is_valid():
    st = ["{","(", ")", "}"]
    status = is_valid(st)
    assert status == True, f"fail the test"
    print("Pass the test")
    
test_is_valid()


def find_max_pair(A:List[Tuple[int, int]], 
                  B:List[Tuple[int, int]], 
                  target: int) -> List[Tuple[int, int]]:
    
    if not A and not B:
        return []
    res = []
    #sort based on second
    A = sorted(A, key=lambda x : x[1])
    B = sorted(B, key=lambda x : x[1])
    
    l = 0; r = len(B)
    
    bestSum = 0
    while l < len(A) and r >= 0:
        if A[l][1] + B[r][1] > target:
            r -= 1
        elif(A[l][1] + B[r][1] > bestSum):
            bestSum = A[l][1] + B[r][1]
            res = [(A[l][0], B[r][0])]
            l += 1
        elif A[l][1] + B[r][1] == target:
            res.append([(A[l][0], B[r][0])])
            l += 1
        else:
            l += 1
    
    return res


def test_find_max_pair():
    A =[(1, 2), (2, 4)]
    B = [(3, 2), (1, 7)]
    target = 10
    res = find_max_pair(A, B, target)
    assert res == [(1, 1)], f"failed the test: {res}"
    print("pass the test")
    
    
def unique_string_detect(st, k: int):
    if len(st) < k:
        return []
    freq = {}
    buf = []
    res = []
    #init the window
    for j in range(k):
        buf.append(st[j])
        freq[st[j]] = freq.get(st[j], 0) + 1
    status = True
    for key in freq:
        status = status and freq[key] == 1
    if status == True:
        res.append(buf)

    for i in range(1, len(st)):
        buf = []
        status = True
        #check the first
        freq[st[i-1]] = freq.get(st[i-1], 0) - 1
        if freq[st[i-1]] == 0:
            del freq[st[i-1]]
        #add the last
        if i + k >= len(st):
            break
        freq[st[i + k]] = freq.get(st[i + k], 0) + 1
        for j in range(i, i + k):
            buf.append(st[j])
        status = True
        for key in freq:
            status = status and freq[key] == 1
        if status == True:
            res.append(buf)
            
    return res
         
def unique_string_detect1(st:str, k:int) -> List[str]:
    if k ==0 or len(st) < k:
        return []
    l = 0
    res = set()
    freq={}
    for r, c in enumerate(st):
        freq[c] = freq.get(c,0) + 1
        if r - l + 1 == k and len(freq) == k:
            res.add(st[l:r+1])
        if r - l + 1 >= k:
            freq[st[l]] = freq.get(st[l], 0) - 1
            if(freq[st[l]]<=0):
                del freq[st[l]]
            l += 1
    return list(res)
    
def test_unique_string_detect():
    st = "aaabccdafk"
    k = 3
    res = unique_string_detect1(st, k)
    print(res)
            
test_unique_string_detect()

def find_k_nearest(pts: List[Tuple[float, float]], k: int) -> List[Tuple[float, float]]:
    if k == 0 or len(pts) < k:
        return []
    tmp = sorted(pts, key=lambda pt:pt[0]**2 + pt[1]**2) 
    return tmp[:k]

def test_find_k_nearest():
    pts = [(0, 0), (10,2), (2,3), (2,5), (1,1)]
    output = find_k_nearest(pts,2)
    expected = set([(0,0), (1,1)])
    assert expected == set(output), f'test was failed {output}'
    print("test was passed")
    
test_find_k_nearest()

from collections import deque
def bfs(A) -> int:
    # using a deque (fifo to store new explor)
    # using a set to keep all visited sites

    m = len(A)
    n = len(A[0])
    qu = deque() #will keep ij and d
    visited = set() #only ij
    end = set()
    for i in range(m):
        for j in range(n):
            c = A[i][j]
            if c == 'D':
                #add into end
                end.add((i, j))
            elif c == 'S':
                #enqueue
                qu.append(((i, j), 0))
                visited.add((i,j))
            elif c == 'O':
                pass
            else:
                ValueError()
        
    #now start search, every time expand 1 and update path point
    while(qu):
        #start explore
        node = qu.popleft()
        # explore
        r = node[0][0]
        c = node[0][1]
        #don't do truncation
        up_idx = (min(m-1, r + 1) , c)
        up = A[min(m-1, r + 1)][c]
        down_idx = (max(0, r-1), c)
        down = A[max(0, r-1)][c]
        left_idx = (r, max(0, c-1))
        left = A[r][max(0, c-1)]
        right_idx = (r, min(n-1, c+1))
        right = A[r][min(n-1, c+1)]
        neighbor = [[up, up_idx],[down, down_idx], [left, left_idx], [right, right_idx]]
        for cu, idx in neighbor:
            if idx not in visited and cu != 'X':
                #add in visited
                visited.add(idx)
                qu.append((idx, node[1]+1))
                if idx in end:
                    return node[1]+1
    return -1
        
def test_bfs():
    A = [['S', 'O', 'O'],['O', 'O', 'O'], ['O', 'O', 'D']]
    print(bfs(A))
    expected = 4
    assert bfs(A)==expected, f'test was failed'
    print('test was passed')
        
        
test_bfs()      
        
    
    
def vec_self_mul(vec: List) -> List:
    #using prefix / suffix only one access to each element, O(n)
    #first pass
    output_left_right = [1]
    for i in range(len(vec)):
        output_left_right.append(output_left_right[-1] * vec[i])
        
    output_right_left = [1]
    for i in range(len(vec)-1, -1, -1):
        output_right_left.append(output_right_left[-1]*vec[i])
    
    output_right_left =  output_right_left[::-1] #start omit, end omit, using step -1
    return [output_left_right[i] * output_right_left[i+1] for i in range(len(vec))]

def test_vec_self_mul():
    vec = [1, 2, 3, 4]
    expected = [24, 12, 8, 6]
    res = vec_self_mul(vec)
    print(res)
    
test_vec_self_mul()
    

            
    
def remove_overlap(vec: List) -> List:
    vec = sorted(vec, key=lambda x: x[1])
    p = 0
    while(p < len(vec)):
        nextP = p + 1
        if nextP >= len(vec):
            return vec
        next = vec[nextP]
        curr = vec[p]
        if next[0] < curr[1]:
            vec.pop(nextP)
        else:
            p = p + 1
    return vec

def test_remove_overlap():
    vec = [[1, 3], [2, 4], [2, 3], [5, 6]]
    print(remove_overlap(vec))
    
test_remove_overlap()


def subarray_sum_k(A: List, k: int) -> int:
    res = 0
    n = len(A)
    seen = {0:1}
    runSum = 0
    for i in range(n):
        c = A[i]
        runSum = runSum + c
        seen[runSum] = seen.get(runSum, 0) + 1
        res += seen.get(runSum - k, 0)
    return res

def test_subarray_sum_k():
    k = 10
    A = [1, 4, 5, 1, 4, -1, 1]
    expected = 3
    print(subarray_sum_k(A, k))

test_subarray_sum_k()

import heapq
def minimum_spending_tree(uvw:List[Tuple[int,int,int]]) -> int:
    start = uvw[0][0] #list of tuple
    hq = [(0, start)] #start from weight 0 and any start point
    visited = set()
    total_cost = 0
    n = len(uvw)
    while hq and len(visited) < n:
        w, cnode = heapq.heappop(hq)
        #check connection
        if cnode in visited:
            continue
        visited.add(cnode)
        total_cost += w
        for u, v, w2 in uvw:
            if u ==cnode and v not in visited:
                heapq.heappush(hq, (w2, v))
            elif v == cnode and u not in visited:
                heapq.heappush(hq, (w2, u))
    return total_cost if len(visited) == n else -1


def group_anagram(st:List[str]) -> List[List[str]]:
    dic={}
    for c in st:
        key = ''.join(sorted(c))
        if key in dic:
            dic[key].append(c)
        else:
            dic[key] = [c] 
    
    return list(dic.values())

def majority_element(vec: List[int]) -> int:
    dic = {}
    n = len(vec)
    for c in vec:
        dic[c] = dic.get(c, 0) + 1
        if dic[c] > n/2:
            return c
    
    return -1


def minimum_remove(vec: List[List[int, int]]) -> List[List[int, int]]:
    vec = sorted(vec, key=lambda x:x[1])
    res = []
    for i in range(len(vec)):
        c = vec(i)
        if i + 1 >= len(vec):
            break
        next = vec[i + 1]
        if c[1] > next[0]:
            res.append(next)
    return res

def solve_min_rooms(vec: List[List[int, int]]) -> int:
    needed = 0
    overlap = vec
    while(overlap):
        overlap = minimum_remove(vec)
        if overlap:
            needed += 1
    return needed


def top_k_words(vec:List[str], k:int) -> List[str]:
    dic = {}
    for c in vec:
        dic[c] = dic.get(c, 0) + 1
    
    sorted_word= sorted(dic, key=lambda k: (-dic[k], k)) #sort only sort key and iterater only on key
    return sorted_word[:k] 

def find_nearest_city(cities: List[Tuple[str, int, int]], query: str) -> str:
    cities1 = sorted(cities, key=lambda x: (x[1], x[2], x[0])) #cloest in terms of y on the same x
    cities2 = sorted(cities, key=lambda x: (x[2], x[1], x[0])) #closet in terms of x on the same y
    
    candidates = []
    for i, city in enumerate(cities1):
        if city[0] == query:
            p = i
            searchIdx = [-1, 1] 
            for k in range(0, len(searchIdx)):
                pN = p + searchIdx[k]
                if pN >= len(cities1) or pN < 0:
                    continue
                elif city[1] == cities1[pN][1]:
                    candidates.append(cities1[pN])

    for i, city in enumerate(cities2):
        if city[0] == query:
            p = i
            searchIdx = [-1, 1] 
            for k in range(0, len(searchIdx)):
                pN = p + searchIdx[k]
                if pN >= len(cities2) or pN < 0:
                    continue
                elif city[2] == cities2[pN][2]:
                    candidates.append(cities1[pN])  
    cx, cy, cn = cities2[p][1], cities2[p][2], cities2[p][0]             
    best = ""
    best_dist = float("inf")
    for name, x, y in candidates:
        dist = abs(x - cx) + abs(y - cy)
        if dist < best_dist or dist == best_dist and name < best:
            best = name
    return best
    
                    
                 
        