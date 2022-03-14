
# Find the intersection point of the two runners.

nums = [1,3,4,2,3,3]
sum = 0
for n in nums:
    sum += n
print(sum - (len(nums) * (len(nums) + 1)) / 2 + len(nums))

tortoise = hare = nums[0]
while True:
    tortoise = nums[tortoise]
    hare = nums[nums[hare]]
    if tortoise == hare:
        break

# Find the "entrance" to the cycle.
tortoise = nums[0]
while tortoise != hare:
    tortoise = nums[tortoise]
    hare = nums[hare]

print(hare)