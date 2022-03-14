"""
322
Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
"""
coins = [1,2,5]
amount = 3

counts = [0] * (amount+1)

for val in range(1,amount+1):
    count = []
    for c in coins:
        if val < c:
            counts[val] = -1
        elif val == c:
            count.append(counts[val - c] + 1)
        elif val > c:
            if counts[val - c] == -1:
                counts[val] = -1
            else:
                count.append(counts[val - c] + 1)
    if not count:
        pass
    else:
        counts[val] = min(count)
print(counts[-1])

