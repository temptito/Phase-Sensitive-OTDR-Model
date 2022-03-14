"""In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

The town judge trusts nobody.
Everybody (except for the town judge) trusts the town judge.
There is exactly one person that satisfies properties 1 and 2.
You are given an array trust where trust[i] = [ai, bi] representing that the person labeled ai trusts the person labeled bi.

Return the label of the town judge if the town judge exists and can be identified, or return -1 otherwise."""

n = 3
trust = [[1,3],[2,3]]

men = [0] * n
prob_judge = [0] * n
for t in trust:
    men[t[0] - 1] = -1
    prob_judge[t[1]-1] +=1

judge = -1
for ind in range(len(men)):
    if (men[ind] == 0) and (prob_judge[ind] == n-1):
        judge = ind + 1
print(judge)
