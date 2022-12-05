### leetcode热门题（501-）

#### 518 零钱兑换二（完全背包）

给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。

请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。

假设每一种面额的硬币有无限个。 

题目数据保证结果符合 32 位带符号整数。

完全背包和01背包的主要区别在于如果选择第i个硬币的话dp\[i][j]要加上的是dp\[i][j - coins[i - 1]]还是dp\[i - 1][j - coins[i - 1]]，完全背包是前面的，因为每种硬币都有无限个。

优化前和优化后的代码：

```java
// class Solution {
//     public int change(int amount, int[] coins) {
//         int len = coins.length;
//         int[][] dp = new int[len + 1][amount + 1];
//         for(int i = 0; i <= len; i++){
//             dp[i][0] = 1;
//         }
//         for(int i = 1; i <= len; i++){
//             for(int j = 1; j <= amount; j++){
//                 dp[i][j] = dp[i - 1][j];
//                 if(coins[i - 1] <= j){
//                     dp[i][j] += dp[i][j - coins[i - 1]];
//                 }
//             }
//         }
//         return dp[len][amount];
//     }   
// }
class Solution {
    public int change(int amount, int[] coins) {
        int len = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        
        for(int i = 1; i <= len; i++){
            for(int j = 1; j <= amount; j++){
                if(coins[i - 1] <= j){
                    dp[j] += dp[j - coins[i - 1]];
                }
            }
        }
        return dp[amount];
    }   
}
```



#### 543 二叉树的直径（类似第124题）

给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过也可能不穿过根结点。

参考的124题思路写的，递归函数的返回值表示经过当前结点的最大路径长度，比较难想的是如果结点为null怎么办，后来发现null的时候返回-1，就能在结点为叶子结点的时候返回0，正好不需要额外的特判

```java
class Solution {
    int max = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        diameterGain(root);
        return max;
    }
    int diameterGain(TreeNode root){
        if(root == null) return -1;//一个小trick
        int left = diameterGain(root.left);
        int right = diameterGain(root.right);

        max = Math.max(max, left + right + 2);
        return Math.max(left, right) + 1;
    }
}
```



#### 547 省份数量（DFS）

有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected\[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected\[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        boolean[] cities = new boolean[n];
        int count = 0;
        for(int i = 0; i < n; i++){
            if(!cities[i]){
                dfs(cities, i, isConnected);
                count++;
            }
        }
        return count;
    }

    void dfs(boolean[] cities, int cur, int[][] isConnected){
        cities[cur] = true;
        for(int i = 0; i < isConnected.length; i++){
            if(isConnected[cur][i] == 1 && cur != i && !cities[i]){
                dfs(cities, i, isConnected);
            }
        }
    }
}
```



#### 560 和为k的子数组（前缀和）

给你一个整数数组 nums 和一个整数 k ，请你统计并返回该数组中和为 k 的连续子数组的个数。

示例 1：

```
输入：nums = [1,1,1], k = 2
输出：2
```

示例 2：

```
输入：nums = [1,2,3], k = 3
输出：2
```


提示：

+ 1 <= nums.length <= 2 * 104
+ -1000 <= nums[i] <= 1000
+ -107 <= k <= 107

看了题解写的，有想到要用前缀和以及哈希表记录，但是没想到处理的顺序，一开始想的是从下标i开始往后有没有比它大k的，这样顶多只能计算到1个，然而可能有多个。

看了题解才知道，哈希表的key是前缀和，value是这个前缀和出现的次数，以及计算的时候并不是哈希表构建完成，而是一边遍历一边计算，这样就可以保证计算的是这个位置前面的。

> 考虑以 i 结尾的和为 k 的连续子数组个数时只要统计有多少个前缀和为 pre[i]-k 的 pre[j] 即可

```java
class Solution {
    public int subarraySum(int[] nums, int k) {
        int len = nums.length;
        int sum = 0;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for(int i = 0; i < len; i++){
            sum += nums[i];
            // 这里顺序很重要，得先计算再把sum加入map，否则会多加上自己的次数
            res += map.getOrDefault(sum - k, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }
}
```



#### 567 字符串的排列（滑动窗口）

给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 

换句话说，s1 的排列之一是 s2 的 子串 。

示例 1：

```
输入：s1 = "ab" s2 = "eidbaooo"
输出：true
解释：s2 包含 s1 的排列之一 ("ba").
```

示例 2：

```
输入：s1= "ab" s2 = "eidboaoo"
输出：false
```


提示：

+ 1 <= s1.length, s2.length <= 104
+ s1 和 s2 仅包含小写字母

做了很久的题，其实不难，但是粗心错了两个地方找了半天

```c++
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        unordered_map<char, int> window, need;
        for(char c : s1){
            need[c]++;
        }
        int right = 0, left = 0;
        int valid = 0;
        while(right < s2.size()){
            char cur = s2[right];
            if(need.count(cur)){
                window[cur]++;
                if(need[cur] == window[cur]){
                    valid++;
                }
            }
            if(right - left + 1 == s1.size()){
                // 一个是这里一开始写成了valid == s1.size()了，
                // 这是另一种做法的写法，这里应该和need.size()比
                if(valid == need.size()) return true;
                char del = s2[left];
                if(need.count(del)){
                    // 一个是这里写成了=了，应该是==
                    if(window[del] == need[del]){ 
                        valid--;                        
                    }
                    window[del]--;
                }
                left++;
            }
            right++;
        }
        return false;
    }
    
};
```



#### 583 两个字符串的删除操作（类似编辑距离，二维DP）

给定两个单词 *word1* 和 *word2*，找到使得 *word1* 和 *word2* 相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        for(int i = 1; i < len1 + 1; i++){
            dp[i][0] = i;
        }
        for(int j = 0; j < len2 + 1; j++){
            dp[0][j] = j;
        }
        
        for(int i = 1; i < len1 + 1; i++){
            for(int j = 1; j < len2 + 1; j++){
                if(word1.charAt(i - 1) != word2.charAt(j - 1)){
                    dp[i][j] = 1 + Math.min(dp[i][j - 1], dp[i - 1][j]);
                }
                else{
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[len1][len2];
    }
}
```



#### 669 修剪二叉搜索树（递归）

给你二叉搜索树的根节点 root ，同时给定最小边界low 和最大边界 high。通过修剪二叉搜索树，使得所有节点的值在[low, high]中。修剪树 不应该 改变保留在树中的元素的相对结构 (即，如果没有被移除，原有的父代子代关系都应当保留)。 可以证明，存在 唯一的答案 。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

```
输入：root = [3,0,4,null,2,null,null,1], low = 1, high = 3
输出：[3,2,null,1]
```


提示：

树中节点数在范围 [1, 104] 内
0 <= Node.val <= 104
树中每个节点的值都是 唯一 的
题目数据保证输入是一棵有效的二叉搜索树
0 <= low <= high <= 104

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if(root == null) return root;
        else if(root.val <= high && root.val >= low){
            root.left = trimBST(root.left, low, high);
            root.right = trimBST(root.right, low, high);
            return root;
        }
        else if(root.val > high){
            return trimBST(root.left, low, high);
        }
        else{
            return trimBST(root.right, low, high);
        }
    }
}
```



#### 684 冗余连接（并查集）

树可以看成是一个连通且 无环 的 无向 图。

给定往一棵 n 个节点 (节点值 1～n) 的树中添加一条边后的图。添加的边的两个顶点包含在 1 到 n 中间，且这条附加的边不属于树中已存在的边。图的信息记录于长度为 n 的二维数组 edges ，edges[i] = [ai, bi] 表示图中在 ai 和 bi 之间存在一条边。

请找出一条可以删去的边，删除后可使得剩余部分是一个有着 n 个节点的树。如果有多个答案，则返回数组 edges 中最后出现的边。

看了题解做的：

- 如果两个顶点属于相同的连通分量，则说明在遍历到当前的边之前，这两个顶点之间已经连通，因此当前的边导致环出现，为附加的边，将当前的边作为答案返回。

```java
class Solution {
    public int[] findRedundantConnection(int[][] edges) {
        int[] parent = new int[edges.length + 1];
        for(int i = 0; i < parent.length; i++){
            parent[i] = i;
        }
        for(int[] edge: edges){
            if(find(parent, edge[0]) != find(parent, edge[1])){
                union(parent, edge[0], edge[1]);
            }
            else{
                return edge;
            }
        }
        return new int[0];
    }
    void union(int[] parent, int index1, int index2){
        parent[find(parent, index1)] = find(parent, index2);
    }
    int find(int[] parent, int index){
        while(parent[index] != index){
            index = parent[index];
        }
        return index;
    }
}
```

并查集的代码比想的要简单



#### 692 前k个高频单词

给一非空的单词列表，返回前 k 个出现次数最多的单词。

返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率，按字母顺序排序。

示例 1：

```
输入: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出: ["i", "love"]
解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。注意，按字母顺序 "i" 在 "love" 之前。
```


示例 2：

```
输入: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
输出: ["the", "is", "sunny", "day"]
解析: "the", "is", "sunny" 和 "day" 是出现次数最多的四个单词，出现次数依次为 4, 3, 2 和 1 次。
```


注意：

+ 假定 k 总为有效值， 1 ≤ k ≤ 集合元素数。
+ 输入的单词均由小写字母组成。


扩展练习：

+ 尝试以 O(n log k) 时间复杂度和 O(n) 空间复杂度解决。

首先，看到按单词出现频率排序就知道应该要用一个map存单词和它的频率，接下来需要排序，看到是选前k个就想到可以维护一个大小为k的堆（小根堆，优先把小的出栈），所以可以选择优先级队列（按频率从小到大，频率相同的字典序从大到小。

入队出队也很简单，直接入队，看队列的size有没有大于k决定要不要poll

最后需要再reverse，因为poll出来顺序是反的

需要额外注意的是频率相同的单词要按字母序，所以是用str1.compareTo(str2)

```java
class Solution {
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        for(int i = 0; i < words.length; i++){
            map.put(words[i], map.getOrDefault(words[i], 0) + 1);
        }
        Queue<String> queue = new PriorityQueue<>(new Comparator<String>(){
            public int compare(String s1, String s2){
                if(map.get(s1) != map.get(s2)){
                    return map.get(s1) - map.get(s2);
                }
                else{
                    return s2.compareTo(s1);
                }
            }
        });
        for(String s : map.keySet()){
            queue.add(s);
            if(queue.size() > k) queue.poll();
        }
        List<String> list = new ArrayList<>();
        while(!queue.isEmpty()){
            list.add(queue.poll());
        }
        Collections.reverse(list);
        return list;
    }
}
```



#### 704 二分查找

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        int mid = 0;
        while(left <= right){
            mid = left + (right - left) / 2;
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return -1;
    }
};
```



#### 713 乘积小于K的子数组（有点特别的滑动窗口）

给定一个正整数数组 `nums`和整数 `k` 。

请找出该数组内乘积小于 `k` 的连续的子数组的个数。

示例 1:

```
输入: nums = [10,5,2,6], k = 100
输出: 8
解释: 8个乘积小于100的子数组分别为: [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]。
需要注意的是 [10,5,2] 并不是乘积小于100的子数组。
```

示例 2:

```
输入: nums = [1,2,3], k = 0
输出: 0
```


提示: 

1 <= nums.length <= 3 * 104
1 <= nums[i] <= 1000
0 <= k <= 10^6

一开始没想到滑动窗口，只写出了一个O(N^2)的解法

看了题解两次才弄懂。重点是res增加的时机，是while循环出来之后，这时从left乘到right的乘积是小于k的，说明这个范围以right为右端点的子数组都符合要求，所以加上right - left + 1，加一是因为长度为1的子数组也算。**总结一下就是right不断右移，每次都计算以right为右端点符合要求的有几个**

```java
class Solution {
    public int numSubarrayProductLessThanK(int[] nums, int k) {
        if(k <= 1) return 0;
        int res = 0;
        int left = 0, right = 0;
        int mul = 1; 
        while(right < nums.length){
            mul *= nums[right];     
            while(mul >= k){
                mul /= nums[left];
                left++;
            }
            res += (right - left + 1);
            right++;
        }
        return res;
    }
}
```



#### 724 寻找数组的中心下标（前缀和）

数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。

如果数组不存在中心下标，返回 -1 。如果数组有多个中心下标，应该返回最靠近左边的那一个。

从[宫水三叶的刷题日记](https://mp.weixin.qq.com/s/Fvknm_kADTIMpSuWMnTvbA) 看到的，里面有三种做法，选择了最简单的一个，本质上还是前缀和

```java
class Solution {
    public int pivotIndex(int[] nums) {
        int sum = 0;
        for(int i = 0; i < nums.length; i++){
            sum += nums[i];
        }
        int total = 0;
        for(int i = 0; i < nums.length; i++){
            total += nums[i];
            if(sum - total == total - nums[i]){
                return i;
            }
        }
        return -1;
    }
}
```

前缀和模板

```java
class Solution {
    public void func(int[] nums) {
        int n = nums.length;
        int[] sum = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            sum[i] = sum[i - 1] + nums[i - 1];
        }
    }
}
```



#### 739 每日温度（单调递减栈）

给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指在第 i 天之后，才会有更高的温度。如果气温在这之后都不会升高，请在该位置用 0 来代替。

```java
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[temperatures.length];
        for(int i = 0; i < temperatures.length; i++){
            while(!stack.isEmpty() && temperatures[stack.peek()] < temperatures[i]){
                int x = stack.pop();
                res[x] = i - x;
            }
            stack.push(i);
        }
        // while(!stack.isEmpty()){
        //     int x = stack.pop();
        //     res[x] = 0;
        // }
        return res;
    }
}
```

go

```go
func dailyTemperatures(temperatures []int) []int {
    length := len(temperatures)
    res := make([]int, length)
    stack := []int{}
    for i := 0; i < length; i++ {
        var x int
        for len(stack) != 0 && temperatures[stack[len(stack) - 1]] < temperatures[i] {
            x = stack[len(stack) - 1]
            res[x] = i - x
            stack = stack[:len(stack) - 1]
        }
        stack = append(stack, i)
    }
    return res
}
```



#### 752 打开转盘锁（BFS）

你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

刚看到题目就很懵，感觉很难，看了下面这个题解才知道其实就是一个广度优先遍历的题

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247485134&idx=1&sn=fd345f8a93dc4444bcc65c57bb46fc35&chksm=9bd7f8c6aca071d04c4d383f96f2b567ad44dc3e67d1c3926ec92d6a3bcc3273de138b36a0d9&scene=21#wechat_redirect

题解写得挺好的，一开始就要做到全对很难，可以先写出简化版的做法，就是先不考虑deadends，直接求从0000到target的次数，这里相当于是最短距离，每组数字相当于图上的一个点，每个点都相邻8个点，就是4个位上的数字可以加一或者减一。从0000开始BFS遍历图，使用step变量记录步数，遇到target提前返回。这些做法和二叉树层次遍历一样。由于是图，所以要比二叉树多加一个visited来判断某个点是否走过了。

现在只剩下不能踩到deadends了，其实这就相当于图里的这几个点不能访问，可以用一个变量来记录，更优的做法是一开始就把这些点加进visited里，后面就不会访问到了。

另外需要注意0000一开始加入了queue，也要加入visited，以及如果0000在deadends里的话直接返回-1

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        Set<String> visited = new HashSet<>();
        for(int i = 0; i < deadends.length; i++){
            visited.add(deadends[i]);
        }
        if(visited.contains("0000")) return -1;
        Queue<String> queue = new LinkedList<>();
        queue.add("0000");
        visited.add("0000");
        int steps = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                String cur = queue.poll();
                if(cur.equals(target)) return steps;
                for(int j = 0; j < 4; j++){
                    String up = up(cur, j);
                    if(!visited.contains(up)){
                        visited.add(up);
                        queue.add(up);
                    }
                    String down = down(cur, j);
                    if(!visited.contains(down)){
                        visited.add(down);
                        queue.add(down);
                    }
                }
            }
            steps++;
        }
        return -1;
    }
    String up(String cur, int i){
        char[] chars = cur.toCharArray();
        if(chars[i] == '9') chars[i] = '0';
        else chars[i] = (char)(chars[i] + 1);    
        return new String(chars);
    }
    String down(String cur, int i){
        char[] chars = cur.toCharArray();
        if(chars[i] == '0') chars[i] = '9';
        else chars[i] = (char)(chars[i] - 1);
        return new String(chars);
    }
}
```



#### 781 森林里的兔子

森林中有未知数量的兔子。提问其中若干只兔子 "还有多少只兔子与你（指被提问的兔子）颜色相同?" ，将答案收集到一个整数数组 answers 中，其中 answers[i] 是第 i 只兔子的回答。

给你数组 answers ，返回森林中兔子的最少数量。

示例 1：

```
输入：answers = [1,1,2]
输出：5
解释：
两只回答了 "1" 的兔子可能有相同的颜色，设为红色。 
之后回答了 "2" 的兔子不会是红色，否则他们的回答会相互矛盾。
设回答了 "2" 的兔子为蓝色。 
此外，森林中还应有另外 2 只蓝色兔子的回答没有包含在数组中。 
因此森林中兔子的最少数量是 5 只：3 只回答的和 2 只没有回答的。
```

示例 2：

```
输入：answers = [10,10,10]
输出：11
```


提示：

1 <= answers.length <= 1000
0 <= answers[i] < 1000

这道题不难，找到规律就好做。回答1一定是有两只，尽可能重叠，所以两个回答1最少是两只，3个回答1最少是4只

然后这道题由于数据的范围是有限的，可以用vector来取代map

```c++
class Solution {
public:
    int numRabbits(vector<int>& answers) {
        int n = 1001;
        vector<int> counts(n,0);
        for(int i = 0; i < answers.size(); i++){
            int answer = answers[i];
            counts[answer]++;
        }
        int res = 0;
        for(int i = 0; i < n; i++){
            int per = i + 1;
            res += per * (counts[i] / per);
            if(counts[i] % per != 0) res += per; 
        }
        return res;
    }
};
```



#### 797 所有可能的路径（DFS）

给你一个有 n 个节点的 有向无环图（DAG），请你找出所有从节点 0 到节点 n-1 的路径并输出（不要求按特定顺序）

 graph[i] 是一个从节点 i 可以访问的所有节点的列表（即从节点 i 到节点 graph\[i][j]存在一条有向边）。



因为是有向无环图，所以不需要visited数组

```java
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        dfs(res, list, graph, 0, graph.length - 1);
        return res;
    }
    void dfs(List<List<Integer>> res, List<Integer> list, int[][] graph, int start, int target){
        if(start == target){
            list.add(start);
            res.add(new ArrayList<>(list));
            list.remove(list.size() - 1);
            return;
        }
        int[] nodes = graph[start];
        list.add(start);
        for(int node : nodes){
            dfs(res, list, graph, node, target);
        }
        list.remove(list.size() - 1);
    }
}
```



#### 815 公交路线（BFS）

```java
class Solution {
    public int numBusesToDestination(int[][] routes, int source, int target) {
        if(source == target) return 0;
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for(int i = 0; i < routes.length; i++){
            for(int station : routes[i]){
                Set<Integer> set = map.getOrDefault(station, new HashSet<>());
                set.add(i);
                map.put(station, set);
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[routes.length];
        queue.add(source);

        int step = 1;
        while(!queue.isEmpty()){
            int size = queue.size();
            while(size > 0){
                int curStation = queue.poll();
                Set<Integer> buses = map.get(curStation);
                for(int bus : buses){
                    if(visited[bus]) continue;
                    visited[bus] = true;
                    for(int nextStation : routes[bus]){
                        if(nextStation == curStation) continue;
                        if(nextStation == target){
                            return step;
                        }
                        queue.add(nextStation);
                        
                    }
                }
                
                size--;
            }
            step++;
        }
        return -1;
    }
}
```



#### 844 比较含退格的字符串（逆序遍历）

给定 s 和 t 两个字符串，当它们分别被输入到空白的文本编辑器后，如果两者相等，返回 true 。# 代表退格字符。

注意：如果对空文本输入退格字符，文本继续为空。

 

示例 1：

```
输入：s = "ab#c", t = "ad#c"
输出：true
解释：s 和 t 都会变成 "ac"。
```

示例 2：

```
输入：s = "ab##", t = "c#d#"
输出：true
解释：s 和 t 都会变成 ""。
```

示例 3：

```
输入：s = "a#c", t = "b"
输出：false
解释：s 会变成 "c"，但 t 仍然是 "b"。
```


提示：

+ 1 <= s.length, t.length <= 200
+ s 和 t 只含有小写字母以及字符 '#'


进阶：

+ 你可以用 O(n) 的时间复杂度和 O(1) 的空间复杂度解决该问题吗？

这道题其实不难，如果不用O(1)的空间复杂度的话。只要简单的模拟就行了，我用的string builder，也可以用stack。

```java
class Solution {
    public boolean backspaceCompare(String s, String t) {
        StringBuilder strbS = new StringBuilder();
        StringBuilder strbT = new StringBuilder();
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == '#'){
                if(strbS.length() > 0)
                    strbS.deleteCharAt(strbS.length() - 1);
            }
            else{
                strbS.append(s.charAt(i));
            }
        }
        for(int i = 0; i < t.length(); i++){
            if(t.charAt(i) == '#'){
                if(strbT.length() > 0)
                    strbT.deleteCharAt(strbT.length() - 1);
            }
            else{
                strbT.append(t.charAt(i));
            }
        }
        return strbS.toString().equals(strbT.toString());
    }
}
```

如果要用O(1)，就麻烦了，看了题解才知道是从后往前遍历字符串，遇到#就给skip变量加一，遇到其他字符就看看skip是否大于一，大于的话就skip--，看下一个字符，小于的话就break出来和另一个字符串比较

如果两个字符串break出来的时候对应的字符不等，或者一个有字符一个已经没有字符了，就是false。否则就继续，注意在最外层while出来后是return true，对应的是比如ab abc#这种

```java
class Solution {
    public boolean backspaceCompare(String s, String t) {
        int iter1 = s.length() - 1, iter2 = t.length() - 1;
        int skip1 = 0, skip2 = 0;
        while(iter1 >= 0 || iter2 >= 0){
            while(iter1 >= 0){
                if(s.charAt(iter1) == '#'){
                    skip1++;
                }
                else if(skip1 > 0){
                    skip1--;
                }
                else{
                    break;
                }
                iter1--;    
            }
            while(iter2 >= 0){
                if(t.charAt(iter2) == '#'){
                    skip2++;
                }
                else if(skip2 > 0){
                    skip2--;
                }
                else{
                    break;
                }
                iter2--;
            }
            if(iter1 >= 0 && iter2 >= 0){
                if(s.charAt(iter1) != t.charAt(iter2)){
                    return false;
                }
                else{
                    iter1--;
                    iter2--;
                }
            }
            else if(iter1 < 0 || iter2 < 0){
                return true;
            }
            else{
                return false;
            }
        }
        return true;
    }
}
```



#### 870 优势洗牌（田忌赛马，双指针，优先队列）

给定两个大小相等的数组 A 和 B，A 相对于 B 的优势可以用满足 A[i] > B[i] 的索引 i 的数目来描述。

返回 A 的任意排列，使其相对于 B 的优势最大化。

示例 1：

```
输入：A = [2,7,11,15], B = [1,10,4,11]
输出：[2,11,7,15]
```

示例 2：

```
输入：A = [12,24,8,32], B = [13,25,32,11]
输出：[24,32,8,12]
```


提示：

+ 1 <= A.length = B.length <= 10000
+ 0 <= A[i] <= 10^9
+ 0 <= B[i] <= 10^9

大致思路是，先对A和B排序，从最大的开始看，如果当前的A比B大，就设置，如果没有，就选一个最小的A去对应这个B。这时候很重要的是知道我们在处理的B的元素的索引，才能把A放到对应的位置。所以在排序B的时候需要带上B的索引。这里用了一个PriorityQueue<int[]>，int[]存放索引和值

https://leetcode-cn.com/problems/advantage-shuffle/solution/java-pai-xu-shuang-zhi-zhen-by-programme-wszf/ 这个题解写得不错

```java
class Solution {
    public int[] advantageCount(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        PriorityQueue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>(){
            public int compare(int[] o1, int[] o2){
                return o2[1] - o1[1];
            }
        });
        for(int i = 0; i < nums2.length; i++){
            queue.add(new int[]{i, nums2[i]});
        }
        int left = 0, right = nums1.length - 1;
        int[] res = new int[nums1.length];
        while(!queue.isEmpty()){
            int[] cur = queue.poll();
            if(cur[1] < nums1[right]){
                res[cur[0]] = nums1[right];
                --right;
            }
            else{
                res[cur[0]] = nums1[left];
                ++left;
            }
        }
        return res;
    }
}
```



#### 871 最低加油次数（有点难的贪心+优先级队列）

汽车从起点出发驶向目的地，该目的地位于出发位置东面 target 英里处。

沿途有加油站，每个 station[i] 代表一个加油站，它位于出发位置东面 station[i][0] 英里处，并且有 station[i][1] 升汽油。

假设汽车油箱的容量是无限的，其中最初有 startFuel 升燃料。它每行驶 1 英里就会用掉 1 升汽油。

当汽车到达加油站时，它可能停下来加油，将所有汽油从加油站转移到汽车中。

为了到达目的地，汽车所必要的最低加油次数是多少？如果无法到达目的地，则返回 -1 。

注意：如果汽车到达加油站时剩余燃料为 0，它仍然可以在那里加油。如果汽车到达目的地时剩余燃料为 0，仍然认为它已经到达目的地。

示例 1：

```
输入：target = 1, startFuel = 1, stations = []
输出：0
解释：我们可以在不加油的情况下到达目的地。
```

示例 2：

```
输入：target = 100, startFuel = 1, stations = [[10,100]]
输出：-1
解释：我们无法抵达目的地，甚至无法到达第一个加油站。
```

示例 3：

```
输入：target = 100, startFuel = 10, stations = [[10,60],[20,30],[30,30],[60,40]]
输出：2
```

解释：
我们出发时有 10 升燃料。
我们开车来到距起点 10 英里处的加油站，消耗 10 升燃料。将汽油从 0 升加到 60 升。
然后，我们从 10 英里处的加油站开到 60 英里处的加油站（消耗 50 升燃料），
并将汽油从 10 升加到 50 升。然后我们开车抵达目的地。
我们沿途在1两个加油站停靠，所以返回 2 。



有点难想到思路，看了宫水三叶的题解，在她的基础上改了一下。i表示的是加油站的下标，start表示是不是起点，起点的时候queue是empty但是不应该返回-1，loc是车当前能开到的位置。当车还没开到终点的时候，可以从queue里取出一个能加油最多的站加上一次油（这里queue存的就是当前还没用过且可以加的油），res++

比较难的点是循环和几个操作的顺序。后来用了start变量，顺序就都可以了。

第一种写法用了start：

```java
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        Queue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        int loc = startFuel;
        int i = 0;
        int res = 0;
        boolean start = true;
        while(loc < target){
            if(!queue.isEmpty()){
                loc += queue.poll();
                res++;
            } 
            else if(start){
                start = false;
            }
            else {
                return -1;
            }  
            for(; i < stations.length && stations[i][0] <= loc; i++){
                queue.add(stations[i][1]);
            }
        }
        return res;
    }
}
```

第二种写法连start也不需要了，先走到startFuel支持的最远的地方，然后看能不能加油，选一个最大的加油量，计算出新的可以到达的最远的地方，进入下一轮循环；如果不能加油，就说明到不了终点，返回-1。

```java
class Solution {
    public int minRefuelStops(int target, int startFuel, int[][] stations) {
        Queue<Integer> queue = new PriorityQueue<>((a, b) -> b - a);
        int loc = startFuel;
        int i = 0;
        int res = 0;
        while(loc < target){
            for(; i < stations.length && stations[i][0] <= loc; i++){
                queue.add(stations[i][1]);
            }
            if(!queue.isEmpty()){
                loc += queue.poll();
                res++;
            } 
            else {
                return -1;
            } 
        }
        return res;
    }
}
```



#### 875 爱吃香蕉的珂珂（二分查找）

珂珂喜欢吃香蕉。这里有 N 堆香蕉，第 i 堆中有 piles[i] 根香蕉。警卫已经离开了，将在 H 小时后回来。

珂珂可以决定她吃香蕉的速度 K （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 K 根。如果这堆香蕉少于 K 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。  

珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。

返回她可以在 H 小时内吃掉所有香蕉的最小速度 K（K 为整数）。

示例 1：

```
输入: piles = [3,6,7,11], H = 8
输出: 4
```

示例 2：

```
输入: piles = [30,11,23,4,20], H = 5
输出: 30
```

示例 3：

```
输入: piles = [30,11,23,4,20], H = 6
输出: 23
```


提示：

+ 1 <= piles.length <= 10^4
+ piles.length <= H <= 10^9
+ 1 <= piles[i] <= 10^9

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247491336&idx=1&sn=dbcbb07b05ebc7889f944d54d2acebd4&scene=21#wechat_redirect

看了这篇推送写的，光看题目没想到是二分。

重点是要抽象出一个单调函数，把题目化解为求一个单调函数在某个取值范围的下标。

```java
class Solution {
    public int minEatingSpeed(int[] piles, int h) {
        int left = 1, right = (int)1e9;
        while(left < right){
            int mid = left + (right - left) / 2;
            int time = timeNeed(piles, mid);
            if(time == h){
                right = mid;
            }
            else if(time > h){
                left = mid + 1;
            }
            else{
                right = mid;
            }
        }
        return right;
    }
    int timeNeed(int[] piles, int k){
        int res = 0;
        for(int i = 0; i < piles.length; i++){
            if(piles[i] <= k){
                ++res;
            }
            else{
                res += piles[i] / k;
                if(piles[i] % k != 0) ++res;
            }
        }
        return res;
    }
}
```



#### 876 链表的中间节点

给定一个头结点为 `head` 的非空单链表，返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

```java
class Solution {
    public ListNode middleNode(ListNode head) {
        if(head == null || head.next == null) return head;
        ListNode slow = head, fast = head.next;
        while(fast.next != null && fast.next.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow.next;
    }
}
```



#### 879 盈利计划（hard, 三维背包）

集团里有 n 名员工，他们可以完成各种各样的工作创造利润。

第 i 种工作会产生 profit[i] 的利润，它要求 group[i] 名成员共同参与。如果成员参与了其中一项工作，就不能参与另一项工作。

工作的任何至少产生 minProfit 利润的子集称为 盈利计划 。并且工作的成员总数最多为 n 。

有多少种计划可以选择？因为答案很大，所以 返回结果模 10^9 + 7 的值。

这题需要使用三维dp数组，我觉得难点除了dp数组的定义，dp的递推，还有最开始的初始化和最后的累加。最后的累加有点难理解。

```java
class Solution {
    public int profitableSchemes(int n, int minProfit, int[] group, int[] profit) {
        int mod = 1000000007;
        int workNum = group.length;
        int[][][] dp = new int[workNum + 1][n + 1][minProfit + 1];

        dp[0][0][0] = 1;
        for(int i = 1; i <= workNum; i++){
            for(int j = 0; j <= n; j++){
                for(int k = 0; k <= minProfit; k++){
                                       
                    if(j >= group[i - 1]){
                        dp[i][j][k] = dp[i - 1][j][k] + dp[i - 1][j - group[i - 1]][Math.max(0, k - profit[i - 1])];
                        dp[i][j][k] %= mod;
                    }
                    else{
                        dp[i][j][k] = dp[i - 1][j][k];
                    }
                        
                }
            }
        }
        int sum = 0;
        for(int i = 0; i <= n; i++){
            sum = (sum + dp[workNum][i][minProfit]) % mod;
        }
        return sum;
    }
}
```



#### 912 排序数组

利用这道题复习了快排，归并，堆排序

https://leetcode-cn.com/problems/sort-an-array/solution/pai-xu-shu-zu-by-leetcode-solution/

```java
class Solution {
    int[] tmp; // 使用一个全局变量节省空间
    Random random = new Random(); // 使用一个全局变量不用重复new
    public int[] sortArray(int[] nums) {   
        // 深拷贝一下就不会改到原来的数组了
        int[] res = Arrays.copyOf(nums, nums.length);
        // tmp = new int[nums.length];
        // mergeSort(res, 0, res.length - 1);
        
        // quickSort(res, 0, res.length - 1);

        heapSort(res);
        return res;
    }
    public void mergeSort(int[] nums, int low, int high){
        if(low == high) return;
        int mid = low + (high - low) / 2;
        mergeSort(nums, low, mid);
        mergeSort(nums, mid + 1, high);

        merge(nums, low, mid, high);
    }
    void merge(int[] nums, int low, int mid, int high){
        int i = low;
        int j = mid + 1;
        int k = low;
        while(i <= mid && j <= high){
            // 注意这里要用小于等于来保证是稳定排序
            if(nums[i] <= nums[j]){
                tmp[k] = nums[i];
                i++;
            }
            else{
                tmp[k] = nums[j];
                j++;                
            }
            k++;
        }
        while(i <= mid){
            tmp[k] = nums[i];
            i++;
            k++;
        }
        while(j <= high){
            tmp[k] = nums[j];
            j++;
            k++;
        }
        // for(i = low; i <= high; i++){
        //     nums[i] = tmp[i];
        // }
        // 这个函数很好用
        // 把tmp里排好序的放回nums
        System.arraycopy(tmp, low, nums, low, high - low + 1);
    }

    public void quickSort(int[] nums, int low, int high){
        // 注意这里是大于等于，不一定是等于，可能是大于
        if(low >= high) return;
        int mid = partition(nums, low, high);
        quickSort(nums, low, mid - 1);
        quickSort(nums, mid + 1, high);
    }

    int partition(int[] nums, int low, int high){
        // 这道题取随机和不取随机时间差很多
        int rand = random.nextInt(high - low + 1) + low;
        swap(nums, rand, low);
        int tmp = nums[low];
        while(low < high){
            while(low < high && nums[high] > tmp){
                high--;
            }
            nums[low] = nums[high];
            while(low < high && nums[low] <= tmp){
                low++;
            }
            nums[high] = nums[low];
        }
        nums[low] = tmp;
        return low;
    }

    void heapSort(int[] nums){
        buildHeap(nums);
        // 注意end的取值
        for(int end = nums.length - 1; end >= 1; end--){
            swap(nums, 0, end);    
            maxHeapify(nums, 0, end);
        }
    }

    void buildHeap(int[] nums){
        // 注意这里是(len-1-1)/2, 相当于是最后一个结点len-1的父结点
        // 父结点计算是(x-1)/2
        for(int i = (nums.length - 2) / 2; i >= 0; i--){
            maxHeapify(nums, i, nums.length);
        }
    }

    // 参数start,end是左闭右开
    void maxHeapify(int[] nums, int start, int end){
        int i = start;
        while(i < end){
            int leftChild = 2 * i + 1;
            int rightChild = 2 * i + 2;
            int large = i;
            if(leftChild < end && nums[leftChild] > nums[i]){
                large = leftChild;
            }
            if(rightChild < end && nums[rightChild] > nums[large]){
                large = rightChild;
            }
            if(large != i){
                swap(nums, i, large);
                i = large;
            }
            else{
                break;
            }

        }
    }

    void swap(int[] nums, int i, int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

https://en.wikipedia.org/wiki/Best,_worst_and_average_case

堆排序建堆的时间复杂度是O(N)，每次调整的复杂度是O(logN)，调整N-1次，所以总时间复杂度是O(NlogN)，空间复杂度是O(1)



#### 986 区间列表的交集（有点难的双指针）

给定两个由一些 闭区间 组成的列表，firstList 和 secondList ，其中 firstList[i] = [starti, endi] 而 secondList[j] = [startj, endj] 。每个区间列表都是成对 不相交 的，并且 已经排序 。

返回这 两个区间列表的交集 。

形式上，闭区间 [a, b]（其中 a <= b）表示实数 x 的集合，而 a <= x <= b 。

两个闭区间的 交集 是一组实数，要么为空集，要么为闭区间。例如，[1, 3] 和 [2, 4] 的交集为 [2, 3] 。

大致的思路是类似归并排序的做法，不断地取出右端点较小的区间，求出它可能的交集，然后移动指针到下一个区间，能这么做是因为右端点最小的区间只会有一个交集，如果有两个，它就不是最小的。所以不停地移除右端点最小的区间，就可以不断地得到交集

```java
class Solution {
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        int iter1 = 0, iter2 = 0;
        List<int[]> list = new ArrayList<>();
        while(iter1 < firstList.length && iter2 < secondList.length){
            if(firstList[iter1][1] < secondList[iter2][1]){
                if(secondList[iter2][0] <= firstList[iter1][1]){
                    list.add(new int[]{Math.max(firstList[iter1][0], secondList[iter2][0]), firstList[iter1][1]});
                }
                iter1++;
            }
            else{
                if(firstList[iter1][0] <= secondList[iter2][1]){
                    list.add(new int[]{Math.max(firstList[iter1][0], secondList[iter2][0]), secondList[iter2][1]});
                }
                iter2++;
            }
        }
        return list.toArray(new int[0][0]);
    }
}
```

看了题解的写法感觉自己写得太傻了

```java
class Solution {
  public int[][] intervalIntersection(int[][] A, int[][] B) {
    List<int[]> ans = new ArrayList();
    int i = 0, j = 0;

    while (i < A.length && j < B.length) {
      // Let's check if A[i] intersects B[j].
      // lo - the startpoint of the intersection
      // hi - the endpoint of the intersection
      int lo = Math.max(A[i][0], B[j][0]);
      int hi = Math.min(A[i][1], B[j][1]);
      if (lo <= hi)
        ans.add(new int[]{lo, hi});

      // Remove the interval with the smallest endpoint
      if (A[i][1] < B[j][1])
        i++;
      else
        j++;
    }

    return ans.toArray(new int[ans.size()][]);
  }
}
```



#### 1011 在D天内送达包裹的能力（二分）

传送带上的包裹必须在 days 天内从一个港口运送到另一个港口。

传送带上的第 i 个包裹的重量为 weights[i]。每一天，我们都会按给出重量（weights）的顺序往传送带上装载包裹。我们装载的重量不会超过船的最大运载重量。

返回能在 days 天内将传送带上的所有包裹送达的船的最低运载能力。

示例 3：

```
输入：weights = [1,2,3,1,1], days = 4
输出：3
解释：
第 1 天：1
第 2 天：2
第 3 天：3
第 4 天：1, 1
```


提示：

1 <= days <= weights.length <= 5 * 104
1 <= weights[i] <= 500

类似875题，重点是求所需时间时内部循环不同，以及二分查找的left, right不同。left是所有包裹中最重的那个，要保证船能装下。right是所有包裹的总重量。

```c++
class Solution {
public:
    int shipWithinDays(vector<int>& weights, int days) {
        int left = 0;
        int right = 0;
        for(int weight : weights){
            left = max(left, weight);
            right += weight;
        }
        while(left < right){
            int mid = left + (right - left) / 2;
            int time = needTime(weights, mid);
            if(time == days){
                right = mid;
            }
            else if(time > days){
                left = mid + 1;
            }
            else{
                right = mid;
            }
        }
        return right;
    }
    int needTime(vector<int>& weights, int k){
        int res = 0;
        for(int i = 0; i < weights.size();){
            int cur = k;
            // 注意这里的循环条件
            while(i < weights.size() && cur >= weights[i]){
                cur -= weights[i];
                ++i;
            }
            ++res;
        }
        return res;
    }
};
```



#### 1063 有效子数组的数目（非严格单调递增栈）

给定一个整数数组 nums ，返回满足下面条件的 非空、连续 子数组的数目：

子数组 是数组的 连续 部分。
子数组最左边的元素不大于子数组中的其他元素 。


示例 1：

```
输入：nums = [1,4,2,5,3]
输出：11
解释：有 11 个有效子数组，分别是：[1],[4],[2],[5],[3],[1,4],[2,5],[1,4,2],[2,5,3],[1,4,2,5],[1,4,2,5,3] 。
```

示例 2：

```
输入：nums = [3,2,1]
输出：3
解释：有 3 个有效子数组，分别是：[3],[2],[1] 。
```

示例 3：

```
输入：nums = [2,2,2]
输出：6
解释：有 6 个有效子数组，分别为是：[2],[2],[2],[2,2],[2,2],[2,2,2] 。
```


提示：

+ 1 <= nums.length <= 5 * 104
+ 0 <= nums[i] <= 105

因为当遇到右侧比自己小的数的时候就可以确定以自己开头的子数组有几个，所以想到用单调递增栈

```java
class Solution {
    public int validSubarrays(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        for(int i = 0; i < nums.length; i++){
            while(!stack.isEmpty() && nums[stack.peek()] > nums[i]){
                res += (i - stack.pop());
            }
            stack.push(i);
        }
        while(!stack.isEmpty()){
            res += (nums.length - stack.pop());
        }
        return res;
    }
}
```



#### 1091 二进制矩阵中的最短路径（BFS）

给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。

二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n - 1)）的路径，该路径同时满足下述要求：

路径途经的所有单元格都的值都是 0 。
路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
畅通路径的长度 是该路径途经的单元格总数。

```java
class Solution {
    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {-1, -1}, {-1, 1}, {1, -1}};

    public int shortestPathBinaryMatrix(int[][] grid) {
        Queue<int[]> queue = new LinkedList<>();
        int n = grid.length;
        if(n <= 0 || grid[0][0] == 1 || grid[n - 1][n - 1] == 1) return -1;
        queue.add(new int[]{0, 0});
        grid[0][0] = 1;
        int step = 1;
        while(!queue.isEmpty()){
            int size = queue.size();    
            while(size != 0){
                int[] position = queue.poll();
                if(position[0] == n - 1 && position[1] == n - 1) return step;
                addNeighbor(position[0], position[1], grid, queue);
                size--;
            }
            step++;
        }
        return -1;
    }
    void addNeighbor(int x, int y, int[][] grid, Queue<int[]> queue){ 
        for(int[] direction : directions){
            int curX = x + direction[0];
            int curY = y + direction[1];
            if(isValid(grid, curX, curY)){
                queue.add(new int[]{curX, curY});
                grid[curX][curY] = 1;
            }
        }
    }
    boolean isValid(int[][] grid, int x, int y){
        if(x < 0 || y < 0 || x >= grid.length || y >= grid[0].length || grid[x][y] == 1){
            return false;
        }
        return true;
    }
}
```



#### 1094 拼车（差分数组）

假设你是一位顺风车司机，车上最初有 capacity 个空座位可以用来载客。由于道路的限制，车 只能 向一个方向行驶（也就是说，不允许掉头或改变方向，你可以将其想象为一个向量）。

这儿有一份乘客行程计划表 trips[][]，其中 trips[i] = [num_passengers, start_location, end_location] 包含了第 i 组乘客的行程信息：

+ 必须接送的乘客数量；
+ 乘客的上车地点；
+ 以及乘客的下车地点。

这些给出的地点位置是从你的 初始 出发位置向前行驶到这些地点所需的距离（它们一定在你的行驶方向上）。

请你根据给出的行程计划表和车子的座位数，来判断你的车是否可以顺利完成接送所有乘客的任务（当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false）。

```java
class Solution {
    public boolean carPooling(int[][] trips, int capacity) {
        int longest = 0;
        for(int i = 0; i < trips.length; i++){
            longest = Math.max(trips[i][2], longest);
        }
        int[] prefix = new int[longest + 1];
        for(int i = 0; i < trips.length; i++){
            prefix[trips[i][1]] += trips[i][0];
            prefix[trips[i][2]] -= trips[i][0];
        }
        int cur = 0;
        for(int i = 0; i < prefix.length; i++){
            cur += prefix[i];
            if(cur > capacity) return false; 
        }
        return true;
    }
}
```



#### 1095 山脉数组中查找目标值（类似162）

给你一个 山脉数组 mountainArr，请你返回能够使得 mountainArr.get(index) 等于 target 最小 的下标 index 值。

如果不存在这样的下标 index，就请返回 -1。

何为山脉数组？如果数组 A 是一个山脉数组的话，那它满足如下条件：

首先，A.length >= 3

其次，在 0 < i < A.length - 1 条件下，存在 i 使得：

+ A[0] < A[1] < ... A[i-1] < A[i]
+ A[i] > A[i+1] > ... > A[A.length - 1]


你将 不能直接访问该山脉数组，必须通过 MountainArray 接口来获取数据：

+ MountainArray.get(k) - 会返回数组中索引为k 的元素（下标从 0 开始）
+ MountainArray.length() - 会返回该数组的长度

**先找到峰值的位置**（思路类似162），然后就可以对峰值的左右两边做二分查找了，最多做三次二分查找

```java
/**
 * // This is MountainArray's API interface.
 * // You should not implement it, or speculate about its implementation
 * interface MountainArray {
 *     public int get(int index) {}
 *     public int length() {}
 * }
 */
 
class Solution {
    public int findInMountainArray(int target, MountainArray mountainArr) {
        int peak = findPeak(mountainArr);
        int res = findTarget(mountainArr, target, 0, peak, true);
        if(res != -1) return res;
        else return findTarget(mountainArr, target, peak + 1, mountainArr.length() - 1, false);
    }
    public int findPeak(MountainArray mountainArr){
        int low = 0, high = mountainArr.length() - 1;
        while(low < high){
            int mid = low + (high - low) / 2;
            int midValue = mountainArr.get(mid);
            int midPlus = mountainArr.get(mid + 1);
            if(mid > 0 && midValue > mountainArr.get(mid - 1) && midValue > midPlus) return mid;
            if(midValue > midPlus) high = mid - 1;
            if(midValue < midPlus) low = mid + 1;
        }
        return low;
    }
    public int findTarget(MountainArray mountainArr, int target, int low, int high, boolean flag){
        while(low <= high){
            int mid = low + (high - low) / 2;
            int midValue = mountainArr.get(mid);
            if(midValue == target) return mid;
            if(midValue < target) {
                if(flag) low = mid + 1;
                else high = mid - 1;
            }
            else{
                if(flag) high = mid - 1;
                else low = mid + 1;
            }
        }
        return -1;
    }
}
```



#### 1114 按序打印（多线程信号量）

保证多线程按序输出

c++:

```c++
#include <semaphore.h>

class Foo {
public:
    sem_t firstJobDone;
    sem_t secondJobDone;
    Foo() {
        sem_init(&firstJobDone, 0, 0);
        sem_init(&secondJobDone, 0, 0);
    }

    void first(function<void()> printFirst) {    
        // printFirst() outputs "first". Do not change or remove this line.
        printFirst();

        sem_post(&firstJobDone);
    }

    void second(function<void()> printSecond) {
        sem_wait(&firstJobDone);
        // printSecond() outputs "second". Do not change or remove this line.
        printSecond();
        sem_post(&secondJobDone);
    }

    void third(function<void()> printThird) {
        sem_wait(&secondJobDone);
        // printThird() outputs "third". Do not change or remove this line.
        printThird();
    }
};
```



#### 1115 交替打印FooBar(信号量)

```c++
#include<semaphore.h>

class FooBar {
private:
    int n;
    sem_t s1;
    sem_t s2;

public:
    FooBar(int n) {
        this->n = n;
        sem_init(&s1, 0, 1); // 这里很重要
        sem_init(&s2, 0, 0);
    }

    void foo(function<void()> printFoo) {
        
        for (int i = 0; i < n; i++) {
            sem_wait(&s1);
        	// printFoo() outputs "foo". Do not change or remove this line.
        	printFoo();
            sem_post(&s2);
        }
    }

    void bar(function<void()> printBar) {
        
        for (int i = 0; i < n; i++) {
            sem_wait(&s2);
        	// printBar() outputs "bar". Do not change or remove this line.
        	printBar();
            sem_post(&s1);
        }
    }
};
```



#### 1143 最长公共子序列（类似编辑距离，二维DP）

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。

例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。
两个字符串的 公共子序列 是这两个字符串所共同拥有的子序列。

dp数组里存的是text1[...i]和text2[...j]的最长公共子序列的长度，如果text1[i] == text2[j]，那长度就是1+dp\[i-1][j-1]。否则就是要么去掉 i 要么去掉 j

```java
class Solution {
    public int longestCommonSubsequence(String text1, String text2) {
        int len1 = text1.length();
        int len2 = text2.length();
        int[][] dp = new int[len1 + 1][len2 + 1];
        for(int i = 1; i <= len1; i++){
            for(int j = 1; j <= len2; j++){
                if(text1.charAt(i - 1) == text2.charAt(j - 1)){
                    dp[i][j] = 1 + dp[i - 1][j - 1];
                }
                else{
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[len1][len2];
    }
}
```



#### 1278 分割回文串三（DP）

给你一个由小写字母组成的字符串 s，和一个整数 k。

请你按下面的要求分割字符串：

首先，你可以将 s 中的部分字符修改为其他的小写英文字母。
接着，你需要把 s 分割成 k 个非空且不相交的子串，并且每个子串都是回文串。
请返回以这种方式分割字符串所需修改的最少字符数。

示例 1：

```
输入：s = "abc", k = 2
输出：1
解释：你可以把字符串分割成 "ab" 和 "c"，并修改 "ab" 中的 1 个字符，将它变成回文串。
```

示例 2：

```
输入：s = "aabbc", k = 3
输出：0
解释：你可以把字符串分割成 "aa"、"bb" 和 "c"，它们都是回文串。
```


提示：

1 <= k <= s.length <= 100
s 中只含有小写英文字母。

dp[i]\[j]表示i个字符分割成k个子串所需修改的最少字符数，要求 i>=j 才有意义，i=0,k!=0以及i!=0,k=0也没有意义。所以初始设置dp数组每个位置的值为Integer.MAX_VALUE，只有dp[0]\[0] = 0。

dp[i]\[j] = min(dp[i0]\[j-1] + cost[i0]\[i-1])

cost[i]\[j]表示从第i个字符修改到第j个字符所需的次数，也可以通过二维dp求得

```java
class Solution {
    public int palindromePartition(String s, int k) {
        int len = s.length();
        int[][] cost = new int[len][len];
        for(int span = 2; span <= len; span++){
            for(int i = 0; i <= len - span; i++){
                int j = i + span - 1;
                cost[i][j] = cost[i + 1][j - 1] + (s.charAt(i) == s.charAt(j) ? 0 : 1);
            }
        }
        int[][] dp = new int[len + 1][k + 1];
        for(int[] nums : dp){
            Arrays.fill(nums, Integer.MAX_VALUE);
        }
        dp[0][0] = 0;
        for(int i = 1; i <= len; i++){
            for(int j = 1; j <= i && j <= k; j++){
                if(j == 1) dp[i][j] = cost[0][i - 1];
                else{
                    for(int i0 = j - 1; i0 < i; i0++){
                        dp[i][j] = Math.min(dp[i][j], dp[i0][j - 1] + cost[i0][i - 1]);
                    }
                }
            }
        }
        return dp[len][k];
    }
}
```



#### 1319 连通网络的操作次数（并查集）

用以太网线缆将 n 台计算机连接成一个网络，计算机的编号从 0 到 n-1。线缆用 connections 表示，其中 connections[i] = [a, b] 连接了计算机 a 和 b。

网络中的任何一台计算机都可以通过网络直接或者间接访问同一个网络中其他任意一台计算机。

给你这个计算机网络的初始布线 connections，你可以拔开任意两台直连计算机之间的线缆，并用它连接一对未直连的计算机。请你计算并返回使所有计算机都连通所需的最少操作次数。如果不可能，则返回 -1 。 

这道题看起来难，其实只需要知道有多少并查集就可以了，并查集的个数其实就是总的元素个数减去union的次数

```java
class Solution {
    public int makeConnected(int n, int[][] connections) {
        if(connections.length < n - 1) return -1;
        int[] parent = new int[n];
        for(int i = 0; i < n; i++){
            parent[i] = i;
        }
        int setCount = n;
        for(int[] connection : connections){
            if(find(parent, connection[0]) != find(parent, connection[1])){
                union(parent, connection[0], connection[1]);
                setCount--;
            }
        }
        return setCount - 1;
    }
    int find(int[] parent, int index){
        while(index != parent[index]){
            index = parent[index];
        }
        return index;
    }
    void union(int[] parent, int x, int y){
        parent[find(parent, x)] = find(parent, y);
    }
}
```



#### 1326 灌溉花园的最少水龙头数目（参考45、55跳跃游戏，贪心）

在 x 轴上有一个一维的花园。花园长度为 n，从点 0 开始，到点 n 结束。

花园里总共有 n + 1 个水龙头，分别位于 [0, 1, ..., n] 。

给你一个整数 n 和一个长度为 n + 1 的整数数组 ranges ，其中 ranges[i] （下标从 0 开始）表示：如果打开点 i 处的水龙头，可以灌溉的区域为 [i -  ranges[i], i + ranges[i]] 。

请你返回可以灌溉整个花园的 最少水龙头数目 。如果花园始终存在无法灌溉到的地方，请你返回 -1 

题解大部分都很难懂，但是都指出了可以参考跳跃游戏的贪心解法，这道题和跳跃游戏的区别是区间的左右端点需要通过中点计算，计算出来后就可以直接套用跳跃游戏的解法了，不断维护一个end变量和max变量，当i走到end时更新end，如果i走出end时还没有到达n就表示无解。

```java
class Solution {
    public int minTaps(int n, int[] ranges) {
        int[] distances = new int[n + 1];
        for(int i = 0; i < ranges.length; i++){
            int left = Math.max(0, i - ranges[i]);
            int right = Math.min(n, i + ranges[i]);
            distances[left] = Math.max(distances[left], right);
        }
        int i = 0;
        int end = 0;
        int max = 0;
        int count = 0;
        while(i <= end){
            max = Math.max(max, distances[i]);
            if(end == i){
                end = max;
                count++;
            }
            if(end == n) return count;
            i++;
        }
        return -1;
    }
}
```



#### 1448 统计二叉树中好节点的数目

给你一棵根为 `root` 的二叉树，请你返回二叉树中好节点的数目。

「好节点」X 定义为：从根到该节点 X 所经过的节点中，没有任何节点的值大于 X 的值。

```java
class Solution {
    public int goodNodes(TreeNode root) {
        if(root == null) return 0;
        return goodNodes(root, root.val);
    }
    int goodNodes(TreeNode root, int min){
        if(root == null) return 0;
        int res = 0;
        if(root.val >= min){
            min = root.val;
            res++;
        } 
        return res + goodNodes(root.left, min) + goodNodes(root.right, min);
    }
}
```



#### 一些见解

+ 如果有一个二维数组，且只能向右向下走，很大概率就是二维DP，dp\[i][j]表示的是到(i,j)这个点的结果
+ 字符串编辑距离，正则匹配，最长公共子序列，二维DP
+ 在树/图中求最短距离，BFS
+ 排列组合，回溯