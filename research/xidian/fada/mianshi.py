# Copyright 2023 Xidian University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# class Node(object):
#     """节点类"""
#     def __init__(self, id=-1, lchild=None, rchild=None):
#         self.id = id
#         self.lchild = lchild
#         self.rchild = rchild
#
#
# class Tree(object):
#     """树类"""
#     def __init__(self, root=None):
#         self.root = root
#
#     def add(self, elem):
#         """为树添加节点"""
#         node = Node(elem)
#         if self.root == None:
#             self.root = node
#         else:
#             queue = []
#             queue.append(self.root)
#             while queue:
#                 cur = queue.pop(0)
#                 if cur.lchild == None:
#                     cur.lchild = node
#                     return
#                 elif cur.rchild == None:
#                     cur.rchild = node
#                     return
#                 else:
#                     queue.append(cur.lchild)
#                     queue.append(cur.rchild)

# nodes = str(input())[1:-1].split(',')
#
# flag = 1
#
# deep = 1
#
# ln = len(nodes)
#
# tmp = []
#
# res = []
#
# for i in range(0, ln):
#
#     if (i + 1) < (2 ** deep) and (i + 1) >= (2 ** (deep - 1)):
#         if nodes[i] != '#':
#             tmp.append(nodes[i])
#     else:
#         if flag == 0:
#             tmp.reverse()
#
#         res.append(tmp)
#         flag = flag ^ 1
#         if nodes[i] != '#':
#             tmp = [nodes[i]]
#         else:
#             tmp = []
#         deep = deep + 1
#
# if len(tmp) > 0:
#     if flag == 0:
#         tmp.reverse()
#
#     res.append(tmp)
#
# print(res)
# (1,2,3,#,#,4,5)


code = str(input())[1:-1]

dp = []

for i in range(0, 100):
    dp.append([0, 0])

dp[0][0] = 1
dp[0][1] = 0
ln = len(code)

for i in range(1, ln):

    if code[i - 1] <= '2' and code[i - 1] > '1':
        if code[i - 1] == '2':
            if code[i] <= '6':
                dp[i][1] = dp[i-1][0]
        else:
            dp[i][1] = dp[i-1][0]
    else:
        dp[i][1] = 0
    dp[i][0] += dp[i-1][0] + dp[i-1][1]
    print(dp[i][0])
print(dp[ln-1][0] + dp[ln-1][1])



