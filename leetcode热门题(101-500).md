### leetcode热门题 （101-500）

#### 101 对称二叉树（递归）

给你一个二叉树的根节点 `root` ， 检查它是否轴对称。

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return isSymmetric(root.left, root.right);
    }
    public boolean isSymmetric(TreeNode root1, TreeNode root2){
        if(root1 == null || root2 == null){
            if(root1 == null && root2 == null) return true;
            else return false;
        }
        if(root1.val != root2.val) return false;
        else return isSymmetric(root1.left, root2.right) && isSymmetric(root1.right, root2.left);
    }
}
```

试试Go

```go
func isSymmetric(root *TreeNode) bool {
    return check(root.Left, root.Right)
}
func check(root1 *TreeNode, root2 *TreeNode) bool {
    if root1 == nil || root2 == nil {
        if root1 == nil && root2 == nil {
            return true
        }
        return false
    }
    return root1.Val == root2.Val && check(root1.Left, root2.Right) && check(root1.Right, root2.Left)
}
```

不用递归的思路

> 初始化时我们把根节点入队两次。每次提取两个结点并比较它们的值（队列中每两个连续的结点应该是相等的，而且它们的子树互为镜像），然后将两个结点的左右子结点按相反的顺序插入队列中。当队列为空时，或者我们检测到树不对称（即从队列中取出两个不相等的连续结点）时，该算法结束。

```go
func isSymmetric(root *TreeNode) bool {
    u, v := root, root
    q := []*TreeNode{}
    q = append(q, u)
    q = append(q, v)
    for len(q) > 0 {
        u, v = q[0], q[1]
        q = q[2:]
        if u == nil && v == nil {
            continue
        }
        if u == nil || v == nil {
            return false
        }
        if u.Val != v.Val {
            return false
        }
        q = append(q, u.Left)
        q = append(q, v.Right)
        q = append(q, u.Right)
        q = append(q, v.Left)
    }
    return true
}
```



#### 102 二叉树的层次遍历

给你二叉树的根节点 `root` ，返回其节点值的 **层序遍历** 。 （即逐层地，从左到右访问所有节点）。

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;

        queue.add(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for(int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                if(cur.left != null) queue.add(cur.left);
                if(cur.right != null) queue.add(cur.right);
                list.add(cur.val);
            }
            res.add(list);
        }
        return res;
    }
}
```



#### 103 二叉树的锯齿形层次遍历

给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

一开始想了好一会儿，后来突然反应过来只是把原本的做法加一个`Collections.reverse(list);`

层次遍历也不是无脑写出来的，需要注意

+ 如果是分层输出的话，需要在里层循环开始前拿到队列的size，这就是这一次层需要输出的个数

```java
class Solution {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        boolean flag = true;
        List<List<Integer>> res = new ArrayList<>();
        if(root == null) return res;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            int cnt = queue.size();
            List<Integer> list = new ArrayList<>();
            while(cnt > 0){ // 这里很重要！需要两重循环
                TreeNode cur = queue.poll();
                if(cur.left != null) queue.add(cur.left);
                if(cur.right != null) queue.add(cur.right);
                list.add(cur.val);
                cnt--;
            }
            if(!flag) Collections.reverse(list);
            res.add(list);
            flag = !flag;
        }
        return res;
    }
}
```



#### 105 从前序遍历和中序遍历构造二叉树

这是我写的

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTree(preorder, inorder, 0, inorder.length - 1, 0, inorder.length - 1);
    }
    TreeNode buildTree(int[] preorder, int[] inorder, int preLeft, int preRight, int inLeft, int inRight){
        if(preLeft > preRight || inLeft > inRight) return null;
        int i = inLeft;
        for(; i <= inRight; i++){
            if(inorder[i] == preorder[preLeft]) break;
        }
        TreeNode root = new TreeNode(preorder[preLeft]);
        root.left = buildTree(preorder, inorder, preLeft + 1, preLeft + i - inLeft, inLeft, i - 1);
        root.right = buildTree(preorder, inorder, preLeft + i - inLeft + 1, preRight, i + 1, inRight);
        return root;
    }
}
```

这是题解写的，用了一个hashmap来快速定位

```java
class Solution {
    private Map<Integer, Integer> indexMap;

    public TreeNode myBuildTree(int[] preorder, int[] inorder, int preorder_left, int preorder_right, int inorder_left, int inorder_right) {
        if (preorder_left > preorder_right) {
            return null;
        }

        // 前序遍历中的第一个节点就是根节点
        int preorder_root = preorder_left;
        // 在中序遍历中定位根节点
        int inorder_root = indexMap.get(preorder[preorder_root]);
        
        // 先把根节点建立出来
        TreeNode root = new TreeNode(preorder[preorder_root]);
        // 得到左子树中的节点数目
        int size_left_subtree = inorder_root - inorder_left;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root.left = myBuildTree(preorder, inorder, preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root.right = myBuildTree(preorder, inorder, preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        // 构造哈希映射，帮助我们快速定位根节点
        indexMap = new HashMap<Integer, Integer>();
        for (int i = 0; i < n; i++) {
            indexMap.put(inorder[i], i);
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }
}
```



#### 110 平衡二叉树

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

> 一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        return height(root) != -1;
    }
    int height(TreeNode root){
        if(root == null) return 0;
        int left = height(root.left);
        if(left == -1) return -1;
        int right = height(root.right);
        if(right == -1) return -1;
        // 下面这句第一次写的时候漏掉了
        if(left - right > 1 || right - left > 1) return -1;
        return Math.max(left, right) + 1;
    }
}
```

**注意** 这道题做了好几次了，但我总是忘记应该怎么同时返回是否平衡和树的高度

**用-1表示不平衡，其余表示平衡以及树的高度**

看了题解才发现还有一种自顶向下的写法，我没有往那边想过。思路就是自然而然的：想知道子树的高度还有子树是否平衡，要怎么用一个函数返回两个信息



#### 111 二叉树的最小深度（BFS）

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

**说明：**叶子节点是指没有子节点的节点。

简单的层次遍历，用depth变量记录深度，遇到叶节点提前返回

```java
class Solution {
    public int minDepth(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null) return 0;
        queue.add(root);
        int depth = 1;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                if(node.left == null && node.right == null) return depth;
                if(node.left != null) queue.add(node.left);
                if(node.right != null) queue.add(node.right);
            }
            depth++;
        }
        return -1;
    }
}
```



#### 112 路径总和（递归）

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。

叶子节点 是指没有子节点的节点。

```java
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null) return false;
        if(root.left == null && root.right == null) return root.val == targetSum;
        return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val);
    }
}
```

go

```go
func hasPathSum(root *TreeNode, targetSum int) bool {
    if root == nil {
        return false
    }
    if root.Left == nil && root.Right == nil {
        return root.Val == targetSum
    }
    return hasPathSum(root.Left, targetSum - root.Val) || hasPathSum(root.Right, targetSum - root.Val)
}
```



#### 113 路径总和二（递归）

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有 从根节点到叶子节点 路径总和等于给定目标和的路径。

叶子节点 是指没有子节点的节点。

```java
class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        pathSum(root, targetSum, res, list);
        return res;
    }
    void pathSum(TreeNode root, int targetSum, List<List<Integer>> res, List<Integer> list){
        if(root == null) return;

        list.add(root.val);
        targetSum -= root.val;
        if(root.left == null && root.right == null && targetSum == 0) {
            res.add(new ArrayList<>(list));// 要在叶子节点的时候做这个操作
            // 这里不能return，必须remove之后才能返回
        }
        else{
            pathSum(root.left, targetSum, res, list);
            pathSum(root.right, targetSum, res, list);
        }       
        list.remove(list.size() - 1);
    }
}
```

go

```go
func pathSum(root *TreeNode, targetSum int) [][]int {
    res := [][]int{}
    list := []int{}
    var dfs func(*TreeNode, int)
    dfs = func(root *TreeNode, targetSum int){
        if root == nil {
            return
        }
        list = append(list, root.Val)
        targetSum -= root.Val
        if root.Left == nil && root.Right == nil && targetSum == 0 {
            res = append(res, append([]int(nil), list...))    
        } else { // else 要放在这行，不能放在下一行
            dfs(root.Left, targetSum)
            dfs(root.Right, targetSum)
        }
        list = list[:len(list) - 1]
    }   
    dfs(root, targetSum)
    return res 
}

```



#### 114 二叉树展开为链表

给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

写了个和题解不太一样的递归

```c++
class Solution {
public:
    void flatten(TreeNode* root) {
        flatten_and_get_tail(root);
    }
    TreeNode* flatten_and_get_tail(TreeNode* root) {
        if(root == nullptr) return nullptr;
        TreeNode* left_tail = flatten_and_get_tail(root->left);
        TreeNode* right_tail = flatten_and_get_tail(root->right);
        if(root->left != nullptr){
            left_tail->right = root->right;
            root->right = root->left;
            root->left = nullptr;
        }
        if(right_tail != nullptr)
            return right_tail;
        else if(left_tail != nullptr)
            return left_tail;
        else
            return root;
    }
};
```



#### 116 填充每个节点的下一个右侧节点指针（可以直接复制117）

给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

```java
class Solution {
    Node nextStart, last;
    public Node connect(Node root) {
        if(root == null) return root;
        
        Node start = root;
        while(start != null){
            nextStart = null;
            last = null;
            // 遍历当前行
            for(Node p = start; p != null; p = p.next){
                if(p.left == null) break;
                handle(p.left);
                handle(p.right);
            }
            start = nextStart;
        }
        return root;
    }

    void handle(Node p){
        if(nextStart == null){
            nextStart = p;
        }
        if(last != null){
            last.next = p;
        }
        last = p;
    }
}
```



#### 117 填充每个节点的下一个右侧节点指针二（借助全局变量实现常量空间）

给定一个二叉树

```java
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}
    
    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
};
```

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

初始状态下，所有 next 指针都被设置为 NULL。

进阶：

你只能使用常量级额外空间。
使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。

这是比较简单的层次遍历做法：

```c++
class Solution {
public:
    Node* connect(Node* root) {
        if(!root) return root;
        queue<Node*> q;
        q.push(root);
        while(!q.empty()){
            int n = q.size();    
            for(int i = 0; i < n; i++){
                Node* cur = q.front();
                q.pop();
                if(i != n - 1) cur->next = q.front();
                
                if(cur->left) q.push(cur->left);
                if(cur->right) q.push(cur->right);
            }
        }
        return root;
    }
};
```

看了题解写的下面这种常量空间做法

start存当前行的第一个node。nextStart存下一行的start，last存前一个node用来把下一行连起来

只要用next连起来了，就不需要用queue来存每一行，只要存链表的头结点就行了

所以遍历当前链表时把下一行的节点连成链表，第一行只有根节点，相当于只有一个node的链表

```java
class Solution {
    Node nextStart, last;
    public Node connect(Node root) {
        if(root == null) return root;
        
        Node start = root;
        while(start != null){
            nextStart = null;
            last = null;
            // 遍历当前行
            for(Node p = start; p != null; p = p.next){
                if(p.left != null) handle(p.left);
                if(p.right != null) handle(p.right);
            }
            start = nextStart;
        }
        return root;
    }

    void handle(Node p){
        if(nextStart == null){
            nextStart = p;
        }
        if(last != null){
            last.next = p;
        }
        last = p;
    }
}
```



#### 121 买卖股票的最佳时机

给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

一次遍历，记录有没有比之前的最小值更小，如果小于，就更新最小值，如果大于，就计算差值，更新利润

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) return 0;
        int res = 0;
        int min = prices[0];
        for(int i = 1; i < prices.length; i++){
            if(prices[i] < min) min = prices[i];
            else res = Math.max(res, prices[i] - min);
        }
        return res;
    }
}
```



#### 124 二叉树中的最大路径和（类似最大连续子序列和）

路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

看了题解才做出来的。像这种返回值和全局变量分别代表不同含义的很容易搞得思路混乱。

递归函数的返回值表示当前结点的最大贡献值，也就是当前结点向下走（向左或右，或者不走），最大的路径

全局变量的值存储的是遍历的过程中所遇到的最大的路径和

也就是说，递归函数一边在计算经过当前结点的最大路径和，一边在计算当前结点的最大贡献值（好给它的父结点使用，父结点不能用子结点的最大路径和，只能选子结点向下走的一条路

```java
class Solution {
    int max;
    public int maxPathSum(TreeNode root) {
        max = root.val;
        maxGain(root);
        return max;
    }
    int maxGain(TreeNode root){
        if(root == null) return 0;
        int left = Math.max(maxGain(root.left), 0);
        int right = Math.max(maxGain(root.right), 0);
        max = Math.max(max, left + root.val + right);
        return root.val + Math.max(left, right);
    }
}
```



#### 128 最长连续序列

给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。 

示例 1：

```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
```

示例 2：

```
输入：nums = [0,3,7,2,5,8,4,6,0,1]
输出：9
```


提示：

+ 0 <= nums.length <= 105
+ -109 <= nums[i] <= 109

第一种思路：先对数组排序，然后遍历一遍就可以知道最长是多长了

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        if(nums.length == 0) return 0;
        Arrays.sort(nums);
        int max = 1;
        int cur = 1;
        for(int i = 1; i < nums.length; i++){
            if(nums[i] == nums[i - 1]) continue;
            if(nums[i] == (nums[i - 1] + 1)) cur++;
            else{
                max = Math.max(max, cur);
                cur = 1;
            } 
        }
        if(cur > 0) max = Math.max(max, cur);
        return max;
    }
}
```

第二种思路：使用set去重，然后遍历set里的数字num，如果num-1不在set里，就从num开始往后找最多可以连续几个，如果num-1在set里，那num已经在之前被算过了，就不用做任何事

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for(int num : nums){
            set.add(num);
        }
        int res = 0;
        int cur = 0;
        for(int num : set){
            if(!set.contains(num - 1)){
                while(set.contains(num)){
                    cur++;
                    num++;
                }
                res = Math.max(res, cur);
                cur = 0;
            }
        }
        return res;
    }
}
```

go没有set，用map[int]bool代替，value为false表示不在set里

```go
func longestConsecutive(nums []int) int {
    numSet := map[int]bool{}
    for _, num := range nums {
        numSet[num] = true
    }
    res := 0
    cur := 0
    
    for num := range numSet {
        if !numSet[num - 1] {
            cur = 1
            for numSet[num + 1] {
                cur++
                num++
            }
            if res < cur {
                res = cur
            }
        }
    }
    return res
}
```



#### 129 求根节点到叶节点数字之和（递归）

给你一个二叉树的根节点 root ，树中每个节点都存放有一个 0 到 9 之间的数字。
每条从根节点到叶节点的路径都代表一个数字：

例如，从根节点到叶节点的路径 1 -> 2 -> 3 表示数字 123 。
计算从根节点到叶节点生成的 所有数字之和 。

叶节点 是指没有子节点的节点。

```java
class Solution {
    int sum = 0;
    public int sumNumbers(TreeNode root) {
        dfs(root, 0);
        return sum;
    }
    void dfs(TreeNode root, int pre){
        if(root == null) return;
        pre = pre * 10 + root.val;
        if(root.left == null && root.right == null){
            sum += pre;
        }
        else{
            dfs(root.left, pre);
            dfs(root.right, pre);
        }
        
    }
}
```

go

```go
func sumNumbers(root *TreeNode) int {
    sum := 0
    var dfs func(*TreeNode, int)
    dfs = func(root *TreeNode, pre int) {
        if root == nil {
            return
        }
        pre = pre * 10 + root.Val
        if root.Left == nil && root.Right == nil {
            sum += pre
        } else {
            dfs(root.Left, pre)
            dfs(root.Right, pre)
        }
    }
    dfs(root, 0)
    return sum 
}
```



#### 130 被围绕的区域（DFS）

给你一个 `m x n` 的矩阵 `board` ，由若干字符 `'X'` 和 `'O'` ，找到所有被 `'X'` 围绕的区域，并将这些区域里所有的 `'O'` 用 `'X'` 填充。

做法类似200题，区别只是dfs的起点变成了矩阵的四个边，还有最后多处理一遍

大致思路是把不需要被X填充的O标记出来，剩下的O就是要被填充的

不会被填充的O都可以从四条边通过DFS找到

```java
class Solution {
    public void solve(char[][] board) {    
        int m = board.length;
        if(m == 0) return;
        int n = board[0].length;
        if(n == 0) return;

        for(int i = 0; i < m; i++){
            dfs(board, i, 0);
            dfs(board, i, n - 1);
        }

        for(int j = 1; j < n - 1; j++){
            dfs(board, 0, j);
            dfs(board, m - 1, j);
        }

        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                if(board[i][j] == 'A'){
                    board[i][j] = 'O';
                }
                else if(board[i][j] == 'O'){
                    board[i][j] = 'X';
                }
            }
        }
    }

    void dfs(char[][] board, int i, int j){
        if(i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != 'O') return;
        board[i][j] = 'A';
        dfs(board, i + 1, j);
        dfs(board, i - 1, j);
        dfs(board, i, j + 1);
        dfs(board, i, j - 1);
    }
}
```



#### 134 加油站（神奇的题）

在一条环路上有 n 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

给定两个整数数组 gas 和 cost ，如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1 。如果存在解，则 保证 它是 唯一 的。

看了题解做的，思路清奇

sum表示的是汽油总共剩余量，如果最后sum < 0，肯定不能环绕一周。

sum可以画出一条线，从不同的地方出发的线是平移的关系，要让这条线上移到每个点都大于等于0，所以要找到这条线的最低点，在代码中对应的是index变量，最后返回index + 1，记得取模

```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int sum = 0;
        int index = 0;
        int min = Integer.MAX_VALUE;
        int n = gas.length;
        for(int i = 0; i < n; i++){
            sum += (gas[i] - cost[i]);
            if(min > sum){
                min = sum;
                index = i;
            }
        }
        return sum < 0 ? -1 : (index + 1) % n;
    }
}
```



#### 135 分发糖果（两次扫描）

n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
相邻两个孩子评分更高的孩子会获得更多的糖果。
请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。

思路挺简单的，从左往右扫描一次，再从右往左扫描一次，就可以了

```java
class Solution {
    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] candy = new int[n];
        for(int i = 1; i < n; i++){
            if(ratings[i] > ratings[i - 1] && candy[i] <= candy[i - 1]){
                candy[i] = candy[i - 1] + 1;
            }
        }
        for(int i = n - 2; i >= 0; i--){
            if(ratings[i] > ratings[i + 1] && candy[i] <= candy[i + 1]){
                candy[i] = candy[i + 1] + 1;
            }
        }
        int res = n;
        for(int i = 0; i < n; i++){
            res += candy[i];
        }
        return res;
    }
}
```

还有一种思路可以不用额外的数组来存放扫描的结果，也不需要两次扫描，就是有点难懂。

举例 1，2，4，3，2，1 

分糖果到第三个人的时候是1，2，3，0，0，0

分糖果到第四个人的时候是1，2，3，1，0，0

分糖果到第五个人的时候是1，2，3，1+1，1，0

分糖果到第六个人的时候是1，2，3+1，2+1，1+1，1

就是需要记下递减序列的长度，给前面的数字加1使得满足递减

```java
class Solution {
    public int candy(int[] ratings) {
        int n = ratings.length;
        int ret = 1;
        int inc = 1, dec = 0, pre = 1;
        for (int i = 1; i < n; i++) {
            if (ratings[i] >= ratings[i - 1]) {
                dec = 0;
                pre = ratings[i] == ratings[i - 1] ? 1 : pre + 1;
                ret += pre;
                inc = pre;
            } else {
                dec++;
                if (dec == inc) {
                    dec++;
                }
                ret += dec;
                pre = 1;
            }
        }
        return ret;
    }
}
```



#### 136 只出现一次的数字（异或）

给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

说明：

你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;  
        for(int i = 0; i < nums.size(); i++){
            res ^= nums[i];
        }
        return res;
    }
};
```



#### 138 复制带随机指针的链表

给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **深拷贝**。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。

```java
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Map<Node, Node> map = new HashMap<>();
        Node newHead = new Node(head.val);
        Node newIter = newHead;
        Node iter = head;
        while(iter.next != null){
            map.put(iter, newIter);
            newIter.next = new Node(iter.next.val);            
            iter = iter.next;
            newIter = newIter.next;
        }
        map.put(iter, newIter);
        iter = head;
        newIter = newHead;
        while(iter != null){
            newIter.random = map.getOrDefault(iter.random, null);
            iter = iter.next;
            newIter = newIter.next;
        }
        return newHead;
    }
}
```



#### 141 环形链表（快慢指针）

如果链表中存在环，则返回 `true` 。 否则，返回 `false` 。

我的写法：

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode nodeFast = head;
        ListNode nodeSlow = head;
        while(nodeFast != null){
            if(nodeSlow == nodeFast.next) return true;
            nodeFast = nodeFast.next;
            if(nodeFast == null) return false;
            nodeFast = nodeFast.next;
            nodeSlow = nodeSlow.next;
        }
        return false;
    }
}
```

题解的写法：

```java
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
}
```



#### 142 环形链表二（记住规律）

给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 `null`。

看了题解写的，重点是如果有环要怎么找到入环的第一个节点，通过计算证明可得，当slow == fast时，让第三个指针从头开始走，slow也继续走，他们俩的交点就是入环的点

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if(head == null) return null;
        ListNode fast = head.next;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            if(fast == slow) break;
            fast = fast.next.next;
            slow = slow.next;
        }
        if(fast == null || fast.next == null) return null;
        ListNode detect = new ListNode();
        detect.next = head;
        while(detect != slow){
            detect = detect.next;
            slow = slow.next;
        }
        return detect;
    }
}
```

另一种稍微有一点点不同的写法如下，这个写法会更简洁，但我还是觉得第一种思路比较顺

```java
public class Solution {
    public ListNode detectCycle(ListNode head) {
        if(head == null) return null;
        ListNode fast = head.next;
        ListNode slow = head;
        while(fast != slow){ // 循环终止条件不一样
            if(fast == null || fast.next == null) return null;
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode detect = new ListNode();
        detect.next = head;
        while(detect != slow){
            detect = detect.next;
            slow = slow.next;
        }
        return detect;
    }
}
```



#### 143 重排链表

给定一个单链表 `L` 的头节点 `head` ，单链表 `L` 表示为：

L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public void reorderList(ListNode head) {
        ListNode middle = getMiddle(head);
        ListNode newHead = reverse(middle);
        merge(head, newHead);
    }
    ListNode getMiddle(ListNode head){
        ListNode slow = head;
        ListNode fast = head.next;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode middle = slow.next;
        slow.next = null;
        return middle;
    }

    ListNode reverse(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode pre = head;
        ListNode cur = head.next;
        ListNode next;
        pre.next = null;
        while(cur != null){
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    void merge(ListNode head1, ListNode head2){
        ListNode iter1 = head1, iter2 = head2;
        ListNode dummyHead = new ListNode();
        ListNode iter = dummyHead;
        while(iter1 != null && iter2 != null){
            iter.next = iter1;
            iter1 = iter1.next;
            iter = iter.next;
            iter.next = iter2;
            iter2 = iter2.next;
            iter = iter.next;
        }
        if(iter1 != null) iter.next = iter1;
    }
}
```



#### 146 LRU缓存（面试高频题）

请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：

+ LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
+ int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
+ void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

自己实现了一个双向双端链表，再利用上HashMap，map的value是链表中的node，抽象出了插入和删除节点这两个函数。需要注意删除节点且删除map中对应item时的顺序！

```java
class ListNode{
    ListNode pre;
    ListNode next;
    int key;
    int value;
    public ListNode(){}
    public ListNode(int key, int value){
        this.key = key;
        this.value = value;
    }
}
class LRUCache {
    Map<Integer, ListNode> map = new HashMap<>();
    int capacity;
    int size = 0;
    ListNode dummyHead, dummyTail;
    public LRUCache(int capacity) {
        dummyHead = new ListNode();
        dummyTail = new ListNode();
        dummyHead.next = dummyTail;
        dummyTail.pre = dummyHead;
        this.capacity = capacity;
    }
    void deleteNode(ListNode node){
        node.next.pre = node.pre;
        node.pre.next = node.next;
    }
    void insertNode(ListNode node){
        node.pre = dummyHead;
        node.next = dummyHead.next;
        dummyHead.next.pre = node;
        dummyHead.next = node;
    }
    public int get(int key) {
        if(!map.containsKey(key)) return -1;
        else{
            ListNode node = map.get(key);    
            deleteNode(node);
            insertNode(node);
            return node.value;
        }
    }
    
    public void put(int key, int value) {
        if(map.containsKey(key)){
            ListNode node = map.get(key);    
            deleteNode(node);
            insertNode(node);
            node.value = value;
            node.key = key;    
        }
        else{
            if(size < capacity){    
                size++;
            }
            else{
                ListNode node = dummyTail.pre; // !!!在这里写出了一个bug，找了特别久没看出来
                deleteNode(node);
                map.remove(node.key);
                // bug的写法：
                // deleteNode(dummyTail.pre);
                // delete之后dummyTail.pre已经变了，再拿去remove就错了
                // map.remove(dummyTail.pre.key);
                
                // 其实把顺序反过来就没事了，我真傻
                // map.remove(dummyTail.pre.key);
                // deleteNode(dummyTail.pre);
            }
            ListNode node = new ListNode(key, value);
            insertNode(node);
            map.put(key, node);
        }
        
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```



#### 148 排序链表（归并）

给你链表的头结点 `head` ，请将其按 **升序** 排列并返回 **排序后的链表** 。

我用了归并排序，比较容易卡住的点是找中点，需要用快慢指针，快指针到链表尾时慢指针正好到中间，要注意快指针和慢指针的初值

```java
class Solution {
    public ListNode sortList(ListNode head){
        // 边界条件：[],[1],[1,2],[1,2,3]
        if(head == null || head.next == null) return head;
        ListNode middle = getMiddle(head);
        head = sortList(head);
        middle = sortList(middle);
        return mergeList(head, middle);
    }

    ListNode getMiddle(ListNode head){
        // 这里需要注意！如果fast = head, 无法处理[1,2]这种情况
        ListNode fast = head.next;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode middle = slow.next;
        slow.next = null;
        return middle;
    }

    ListNode mergeList(ListNode head1, ListNode head2){
        ListNode dummyHead = new ListNode();
        ListNode iter = dummyHead;
        ListNode iter1 = head1;
        ListNode iter2 = head2;
        while(iter1 != null && iter2 != null){
            if(iter1.val <= iter2.val){
                iter.next = iter1;
                iter1 = iter1.next;
            }
            else{
                iter.next = iter2;
                iter2 = iter2.next;
            }
            iter = iter.next;
        }
        if(iter1 != null){
            iter.next = iter1;
        }
        if(iter2 != null){
            iter.next = iter2;
        }
        return dummyHead.next;
    }
}
```



#### 151 翻转字符串里的单词

给你一个字符串 s ，逐个翻转字符串中的所有 单词 。

单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。

请你返回一个翻转 s 中单词顺序并用单个空格相连的字符串。

说明：

- 输入字符串 s 可以在前面、后面或者单词间包含多余的空格。
- 翻转后单词间应当仅用一个空格分隔。
- 翻转后的字符串中不应包含额外的空格。

```
输入：s = "  Bob    Loves  Alice   "
输出："Alice Loves Bob"
```

提示：

+ 1 <= s.length <= 104
+ s 包含英文大小写字母、数字和空格 ' '
+ s 中 至少存在一个 单词


进阶：

+ 请尝试使用 O(1) 额外空间复杂度的原地解法。

原地解法只有c++可以做到，java的string是final型的，不行

最简单的做法是用库函数，缺点是时间会稍微久一点

```java
class Solution {
    public String reverseWords(String s) {
        s = s.trim();
        String[] array = s.split("\\s+");
        List<String> list = Arrays.asList(array);
        Collections.reverse(list);
        return String.join(" ", list);
    }
}
```

我最先想到的做法是下面这个，用StringBuilder存中间结果，从后往前遍历地加上子串

```java
class Solution {
    public String reverseWords(String s) {
        s = s.trim();
        StringBuilder strb = new StringBuilder();
        int right = s.length();
        for(int i = right - 1; i >= 0; i--){
            if(s.charAt(i) == ' '){
                strb.append(s.substring(i + 1, right));
                strb.append(" ");
                while(s.charAt(i) == ' '){
                    i--;
                }
                right = i + 1;
            }
        }
        // 注意！！要把最后一个也加上，因为它前面没有空格了，所以循环没处理到
        strb.append(s.substring(0, right));
        return strb.toString();
    }
}
```

用最省内存，最接近原地处理的思路写的代码如下，因为是java所以并没有达到O(1)

大体的思路是先整个reverse，再对每个单词reverse

```java
class Solution {
    public String reverseWords(String s) {
        StringBuilder sb = trimSpaces(s);

        reverse(sb, 0, sb.length() - 1);

        reverseEachWord(sb);

        return sb.toString();
    }
    StringBuilder trimSpaces(String s){
        StringBuilder res = new StringBuilder();
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) != ' ' || (i > 0 && s.charAt(i - 1) != ' ')){
                res.append(s.charAt(i));
            }
        }
        if(res.charAt(res.length() - 1) == ' '){
            res.deleteCharAt(res.length() - 1);
        }
        return res;
    }
    void reverse(StringBuilder sb, int start, int end){
        while(start < end){
            char tmp = sb.charAt(start);
            sb.setCharAt(start, sb.charAt(end));
            sb.setCharAt(end, tmp);
            end--;
            start++;
        }
    }
    void reverseEachWord(StringBuilder sb){
        int start = 0, end = 0;
        for(int i = 0; i < sb.length(); i++){
            if(sb.charAt(i) == ' '){
                end = i - 1;
                reverse(sb, start, end);
                start = i + 1;
            }
        }
        reverse(sb, start, sb.length() - 1);
    }
}
```



#### 153 寻找旋转排序数组中的最小值（二分查找）

已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

画个图，多想想边界情况，思路还挺好想的。我的做法和题解有一点点不一样，题解是不断和nums[high]比较，我是和nums[0]比较，都行。重点是画图！

简单说下思路：利用和nums[0]比较来判断nums[mid]是在数组的左半边还是右半边，因为最小值是在右半边，所以如果nums[mid]在左半，low就更新为mid+1，因为mid不可能是最终答案，反之把high更新为mid，因为nums[mid]可能是最终答案。

比较容易绕晕的是循环结束条件和最后返回的值，循环应该在low==high的时候结束，这时候答案收束为一个点，就可以返回了

```java
class Solution {
    public int findMin(int[] nums) {
        //边界条件[],[1],[1,2],[2,1]
        int low = 0;
        int high = nums.length - 1;
        if(nums[low] <= nums[high]) return nums[low];//直接解决了[1]和[1,2]这两种
        while(low < high){
            int mid = low + (high - low) / 2;
            if(nums[mid] >= nums[0]){
                low = mid + 1;
            }
            else{
                high = mid;
            }
        }
        return nums[low];
    }
}
```



#### 160 相交链表(不要小瞧这道题)

这是我一开始的做法。是很容易想到的思路，但是粗心了好几个地方，打了好几次log，看了很久才发现

而且这个做法的用时很长，用了78ms，下面那种只用了1ms

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) return null;
        ListNode iterA = headA;
        ListNode iterB = headB;
        int countA = 0, countB = 0;
        while(iterA != null){
            iterA = iterA.next;
            countA++;
        }
        while(iterB != null){
            iterB = iterB.next;
            countB++;
        }
        iterA = headA;
        iterB = headB;
        while(countA > countB){ //这里的while一开始写成了if,找了半天才找出来
            iterA = iterA.next;            
            countA--;
        }
        while(countB > countA){
            iterB = iterB.next;
            countB--;
        }
        System.out.println(countB);
        while(iterA != null && iterB != null){
            if(iterA == iterB) return iterA;
            iterA = iterA.next;
            iterB = iterB.next; //这两句忘记写了导致死循环，也是看了半天才看出来
            System.out.println(1);
        }
        return null;
    }
}
```

> 创建两个指针 pA 和 pB，分别初始化为链表 A 和 B 的头结点。然后让它们向后逐结点遍历。
>
> 当 pA 到达链表的尾部时，将它重定位到链表 B 的头结点 (你没看错，就是链表 B); 类似的，当 pB 到达链表的尾部时，将它重定位到链表 A 的头结点。
>
> 若在某一时刻 pA 和 pB 相遇，则 pA/pB 为相交结点。
>
> 想弄清楚为什么这样可行, 可以考虑以下两个链表: A={1,3,5,7,9,11} 和 B={2,4,9,11}，相交于结点 9。 由于 B.length (=4) < A.length (=6)，pB 比 pA 少经过 22 个结点，会先到达尾部。将 pB 重定向到 A 的头结点，pA 重定向到 B 的头结点后，pB 要比 pA 多走 2 个结点。因此，它们会同时到达交点。
>
> 如果两个链表存在相交，它们末尾的结点必然相同。因此当 pA/pB 到达链表结尾时，记录下链表 A/B 对应的元素。若最后元素不相同，则两个链表不相交。
> 链接：https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/xiang-jiao-lian-biao-by-leetcode/

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null) return null;
        ListNode iterA = headA;
        ListNode iterB = headB;
        boolean changedA = false;
        boolean changedB = false;
        while(true){
            if(iterA == null){
                if(changedA) return null;
                iterA = headB;
                changedA = true;
            } 
            if(iterB == null){
                if(changedB) return null;
                iterB = headA;
                changedB = true;
            } 
            if(iterB == iterA) return iterB;
            iterA = iterA.next;
            iterB = iterB.next;
        }
    }
}
```

上面这个是我写的代码，下面这个是别人写的，下面这个简洁很多

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    if (headA == null || headB == null) return null;
    ListNode pA = headA, pB = headB;
    while (pA != pB) {
        pA = pA == null ? headB : pA.next;
        pB = pB == null ? headA : pB.next;
    }
    return pA;
}
// 因为走的总路程相同，pa一定会等于pb, 不相交的情况下就是同时为null
// 所以只要返回两者相同时的值就行了
```



#### 162 寻找峰值（想了好久的二分）

峰值元素是指其值严格大于左右相邻值的元素。

给你一个整数数组 nums，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 nums[-1] = nums[n] = -∞ 。

你必须实现时间复杂度为 O(log n) 的算法来解决此问题。

看到时间复杂度是log就知道要二分，画图分了7种情况，分类讨论处理low和high的赋值，循环终止条件是low==high

```java
class Solution {
    public int findPeakElement(int[] nums) {
        // 边界情况[],[1],[1,2,3],[3,2,1]
        if(nums.length == 1) return 0;
        if(nums.length > 1){
            if(nums[0] > nums[1]) return 0;
            if(nums[nums.length - 1] > nums[nums.length - 2]) return nums.length - 1;
        }
        int low = 0;
        int high = nums.length - 1;
        while(low < high){
            int mid = low + (high - low) / 2;
            if(mid > 0 && nums[mid] > nums[mid - 1]){
                low = mid;
            }
            if(mid < nums.length - 1 && nums[mid] > nums[mid + 1]){
                high = mid;
            }
            // 后面两个都必须加else
            else if(mid > 0 && nums[mid] < nums[mid - 1]){
                high = mid - 1;
            }
            else if(mid < nums.length - 1 && nums[mid] < nums[mid + 1]){
                low = mid + 1;
            }
        }
        return low;
    }
}
```

看了题解，大致思路和我一样，但是加了辅助函数，就可以不用操心数组越界的情况，这样需要讨论的情况就只有四种：峰值，谷值，低中高，高中低。其中谷值可以和低中高合并。

```java
class Solution {
    public int findPeakElement(int[] nums) {
        int n = nums.length;
        int left = 0, right = n - 1, ans = -1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (compare(nums, mid - 1, mid) < 0 && compare(nums, mid, mid + 1) > 0) {
                ans = mid;
                break;
            }
            if (compare(nums, mid, mid + 1) < 0) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }

    // 辅助函数，输入下标 i，返回一个二元组 (0/1, nums[i])
    // 方便处理 nums[-1] 以及 nums[n] 的边界情况
    public int[] get(int[] nums, int idx) {
        if (idx == -1 || idx == nums.length) {
            return new int[]{0, 0};
        }
        return new int[]{1, nums[idx]};
    }

    public int compare(int[] nums, int idx1, int idx2) {
        int[] num1 = get(nums, idx1);
        int[] num2 = get(nums, idx2);
        if (num1[0] != num2[0]) {
            return num1[0] > num2[0] ? 1 : -1;
        }
        if (num1[1] == num2[1]) {
            return 0;
        }
        return num1[1] > num2[1] ? 1 : -1;
    }
}
```

最后！我参考题解的分类，写出了下面这个简洁的版本！

重点在于不用判断mid+1是否越界，因为mid不可能是high，mid可能是low，假如mid==high，那肯定先有low==high，那就退出循环了

```java
class Solution {
    public int findPeakElement(int[] nums) {
        // 边界情况[],[1],[1,2,3],[3,2,1]
        if(nums.length == 1) return 0;
        if(nums.length > 1){
            if(nums[0] > nums[1]) return 0;
            if(nums[nums.length - 1] > nums[nums.length - 2]) return nums.length - 1;
        }
        int low = 0;
        int high = nums.length - 1;
        while(low < high){
            int mid = low + (high - low) / 2;
            if(mid > 0 && nums[mid] > nums[mid - 1] && nums[mid] > nums[mid + 1]) return mid;
            if(nums[mid] > nums[mid + 1]) high = mid;// 可以改成high = mid - 1; 原因见“后续”
            if(nums[mid] < nums[mid + 1]) low = mid + 1;
        }
        return low;
    }
}
```

**后续**

我发现nums[mid] > nums[mid + 1]的时候，mid也不可能是low，如果mid==low，那会再更前面的特判里被处理掉，没有处理到就说明nums[low]<nums[low+1] && nums[high]<nums[high-1]，所以这个if语句里high可以等于mid-1不用担心越界



#### 198 打家劫舍（线性DP）

你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        dp[1] = nums[0];
        for(int i = 2; i <= n; i++){
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return dp[n];
    }
}
```

可以进一步优化dp数组，其实只要知道前两个值就行了

用go写了优化后的

```go
func rob(nums []int) int {
    pre1 := 0
    pre2 := 0
    cur := 0
    for _, num := range nums {
        cur = max(pre1, pre2 + num)
        pre2 = pre1
        pre1 = cur
    }
    return cur
}

func max(num1 int, num2 int) int {
    if num1 > num2 {
        return num1
    } else {
        return num2
    }
}
```



#### 199 二叉树的右视图（层次遍历）

给定一个二叉树的 **根节点** `root`，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。

层次遍历，找到每一层的最右边的点就行了。时间复杂度O(N)

```java
class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        if(root == null) return res;

        queue.add(root);
        while(!queue.isEmpty()){
            int size = queue.size();
            res.add(queue.peek().val);
            for(int i = 0; i < size; i++){
                TreeNode cur = queue.poll();
                if(cur.right != null) queue.add(cur.right);
                if(cur.left != null) queue.add(cur.left);    
            }
        }
        return res;
    }
}
```



#### 200 岛屿数量（DFS）

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

看题目觉得很难，看了题解才发现只是普通的DFS

主要的思路是在DFS的时候把连着的一块陆地从标记1改为标记2（题解里用的是0，都可以），计算可以有几次DFS，就说明有几个岛

```java
class Solution {
    public int numIslands(char[][] grid) {
        int count = 0;
        for(int i = 0; i < grid.length; i++){
            for(int j = 0; j < grid[0].length; j++){
                if(grid[i][j] == '1'){
                    count++;
                    dfs(grid, i, j);
                }
            }
        }
        return count;
    }
    void dfs(char[][] grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length) return;
        if(grid[i][j] != '1') return;
        grid[i][j] = '2';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
        return;
    }
}
```

[通用解法](https://leetcode-cn.com/problems/number-of-islands/solution/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-/)这个题解写得特别好，非常推荐



#### 206 反转链表（超级经典）

**迭代**

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        // if(head == null) return head; 这行可以不写
        ListNode pre = null;
        ListNode cur = head;
        ListNode next;
        while(cur != null){
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;    
        }
        return pre;
    }
}
```

**递归**

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if(head == null || head.next == null) return head; // 注意！记得判空
        ListNode newHead = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    }
}
```



#### 207 课程表（BFS/DFS 拓扑排序）

你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。

在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。

例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

示例 1：

```
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```

看了题解写的代码，用的是拓扑排序的思路，类似BFS

我觉得比较阻碍我写出来的是用一个List\<List\<Integer>>来存边的信息，以及用一个int数组来存每个点的入度。最后是用一个队列来存入度为0的点（之后我验证了，这里用队列或者栈都行，因为只要进来了，就都是入度为0。

主要逻辑就是，遍历入度为0的点，将它们所指向的点的入度减一，如果减一后等于0，就加入队列中。遍历时累计入度为0的点的个数，如果最后等于总的点的个数，就说明没有环。

**注意！**循环终止条件为队列为空。所以第一遍要先把入度为0的点加进队列里

```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Queue<Integer> queue = new LinkedList<>();
        // Stack<Integer> queue = new Stack<>();
        List<List<Integer>> edges = new ArrayList<>();
        int[] indegree = new int[numCourses];
        for(int i = 0; i < numCourses; i++){
            edges.add(new ArrayList<Integer>());
        }
        for(int i = 0; i < prerequisites.length; i++){
            indegree[prerequisites[i][0]]++;
            edges.get(prerequisites[i][1]).add(prerequisites[i][0]);
        }
        for(int i = 0; i < numCourses; i++){
            if(indegree[i] == 0) queue.add(i);
        }
        int visited = 0;
        while(!queue.isEmpty()){
            int i = queue.poll();
            // int i = queue.pop();
            visited++;
            for(int j = 0; j < edges.get(i).size(); j++){
                indegree[edges.get(i).get(j)]--;
                if(indegree[edges.get(i).get(j)] == 0) {
                    queue.add(edges.get(i).get(j));
                    // queue.push(edges.get(i).get(j));
                }
            }
        }
        return visited == numCourses;
    }
}
```

这是我参考210题解的DFS写法写的，没有真的用到栈，只用了一个int数组标记每个点的状态，当这个点已经在搜索中，却又被另一个点搜索时，就说明有环，当一个点的所有邻接点都访问完后，就把状态改为已完成，这种做法比BFS要快

```java
class Solution {
    // 存储有向图
    List<List<Integer>> edges;
    // 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
    int[] visited;

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<List<Integer>>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<Integer>());
        }
        visited = new int[numCourses];
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
        }
        // 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
        for (int i = 0; i < numCourses; ++i) {
            if (!dfs(i)) {
                return false;
            }
        }
        return true;
    }

    public boolean dfs(int u) {
        if(visited[u] == 2) return true;
        if(visited[u] == 1) return false;
        // 将节点标记为「搜索中」
        visited[u] = 1;
        // 搜索其相邻节点
        // 只要发现有环，立刻停止搜索
        for (int v: edges.get(u)) {
            if(!dfs(v)) return false;    
        }
        // 将节点标记为「已完成」
        visited[u] = 2;
        return true;
    }
}
```



#### 209 长度最小的子数组（滑动窗口）

给定一个含有 n 个正整数的数组和一个正整数 target 。

找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。

```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0, right = 0;
        int sum = 0;
        int res = Integer.MAX_VALUE;
        while(right < nums.length){
            sum += nums[right];
            while(sum >= target){
                res = Math.min(res, right - left + 1);
                sum -= nums[left];
                left++;
            }
            right++;
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }
}
```



#### 210 课程表二（BFS/DFS 拓扑排序）

现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。

```java
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        Queue<Integer> queue = new LinkedList<>();
        List<List<Integer>> edges = new ArrayList<>();
        int[] indegree = new int[numCourses];
        for(int i = 0; i < numCourses; i++){
            edges.add(new ArrayList<Integer>());
        }
        for(int i = 0; i < prerequisites.length; i++){
            indegree[prerequisites[i][0]]++;
            edges.get(prerequisites[i][1]).add(prerequisites[i][0]);
        }
        for(int i = 0; i < numCourses; i++){
            if(indegree[i] == 0) queue.add(i);
        }
        int visited = 0;
        int[] res = new int[numCourses];
        while(!queue.isEmpty()){
            int i = queue.poll();
            res[visited] = i;
            visited++;
            for(int j = 0; j < edges.get(i).size(); j++){
                indegree[edges.get(i).get(j)]--;
                if(indegree[edges.get(i).get(j)] == 0) queue.add(edges.get(i).get(j));
            }
        }
        return visited == numCourses ? res : new int[0];
    }
}
```

参考题解的DFS写法写的代码如下

```java
class Solution {
    // 存储有向图
    List<List<Integer>> edges;
    // 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
    int[] visited;
    // 用数组来模拟栈，下标 n-1 为栈底，0 为栈顶
    int[] result;
    // 栈下标
    int index;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        edges = new ArrayList<List<Integer>>();
        for (int i = 0; i < numCourses; ++i) {
            edges.add(new ArrayList<Integer>());
        }
        visited = new int[numCourses];
        result = new int[numCourses];
        index = numCourses - 1;
        for (int[] info : prerequisites) {
            edges.get(info[1]).add(info[0]);
        }
        // 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
        for (int i = 0; i < numCourses; ++i) {
            if (!dfs(i)) {
                return new int[0];
            }
        }
        // 如果没有环，那么就有拓扑排序
        return result;
    }

    public boolean dfs(int u) {
        if(visited[u] == 2) return true;
        if(visited[u] == 1) return false;
        // 将节点标记为「搜索中」
        visited[u] = 1;
        // 搜索其相邻节点
        // 只要发现有环，立刻停止搜索
        for (int v: edges.get(u)) {
            if (!dfs(v)) {
                return false;
            }
        }
        // 将节点标记为「已完成」
        visited[u] = 2;
        // 将节点入栈
        result[index--] = u;
        return true;
    }
}
```



#### 213 打家劫舍二（环形，用直线的做法算两次）

首尾不能同时抢，那就是分成不可以抢首和不可以抢尾两种，所以只要调用两次之前198题计算直线情况的函数就可以了

```java
class Solution {
    public int rob(int[] nums) {
        int n = nums.length;
        return Math.max(rob(nums, 0, n - 2), rob(nums, 1, n - 1));
    }
    public int rob(int[] nums, int left, int right) {
        if(left > right) return nums[0]; // 注意只有一个房子的情况
        int pre1 = 0, pre2 = 0, cur = 0;
        for(int i = left; i <= right; i++){
            cur = Math.max(pre1, pre2 + nums[i]);
            pre2 = pre1;
            pre1 = cur;
        }
        return cur;
    }
}
```

go

```java
func rob(nums []int) int {
    n := len(nums)
    return max(robInStraight(nums, 0, n - 2), robInStraight(nums, 1, n - 1))
}

func robInStraight(nums []int, left int, right int) int {
    if left > right {
        return nums[0]
    }
    pre1, pre2, cur := 0, 0, 0
    for i := left; i <= right; i++ {
        cur = max(pre1, pre2 + nums[i])
        pre2 = pre1
        pre1 = cur
    }
    return cur
}

func max(x int, y int) int {
    if x > y {
        return x
    } else {
        return y
    }
}
```

+ 用go写代码的时候记得写函数返回值的类型



#### 215 数组中的第K个最大元素（面试高频题）

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `k` 个最大的元素。

请注意，你需要找的是数组排序后的第 `k` 个最大的元素，而不是第 `k` 个不同的元素。

```java
class Solution {
    public int findKthLargest(int[] nums, int k) {
        k = nums.length - k;
        int low = 0, high = nums.length - 1;
        while(low < high){
            int pos = partition(nums, low, high);
            if(pos == k) break;
            else if(pos < k){
                low = pos + 1;
            }
            else{
                high = pos - 1;
            }
        }
        return nums[k];
    }
    private int partition(int[] list, int low, int high){
        int tmp = list[low];
        while(low < high){
            while(low < high && list[high] > tmp){
                high--;
            }
            list[low] = list[high];
            while(low < high && list[low] <= tmp){
                low++;
            }
            list[high] = list[low];
        }
        list[low] = tmp;
        return low;
    }
}
```



#### 222 完全二叉树的节点个数

看了[题解](https://leetcode-cn.com/problems/count-complete-tree-nodes/solution/chang-gui-jie-fa-he-ji-bai-100de-javajie-fa-by-xia/)写的，这个题解写得很好，做法非常简洁

思路很巧妙，把情况分成了左子树满和左子树不满两种，当左子树满的时候可以通过公式直接计算出左子树+根节点的总个数（2^leftHeight），然后递归计算右子树的节点个数。当左子树不满的时候，右子树就是满的，类似的，可以通过公式直接计算出右子树+根节点的总个数，然后递归计算左子树的节点个数。

```java
class Solution {
    public int countNodes(TreeNode root) {
        if(root == null) return 0;
        int left = countLevel(root.left);
        int right = countLevel(root.right);
        if(left > right) return countNodes(root.left) + (1 << right);// 必须加括号
        else return countNodes(root.right) + (1 << left);
    }
    int countLevel(TreeNode root){
        if(root == null) return 0;
        TreeNode iter = root;
        int res = 0;
        while(iter != null){
            res++;
            iter = iter.left;
        }
        return res;
    }
}
```

时间复杂度是O(logn*logn)，空间复杂度O(logn)

在评论区看到了非递归的写法，能更清楚地看出时间复杂度，还不需要递归调用栈的空间

```java
	public int countNodesIteration(TreeNode root){
        if(root == null) return 0;
        TreeNode iter = root;
        int res = 0;
        while(iter != null){
            int left = countLevel(iter.left);
            int right = countLevel(iter.right);
            if(left > right){
                res += (1 << right);
                iter = iter.left;
            }
            else{
                res += (1 << left);
                iter = iter.right;
            }
        }
        return res;
    }
```



#### 224 基本计算器（只有加减括号）

给你一个字符串表达式 s ，请你实现一个基本计算器来计算并返回它的值。

示例 1：

```
输入：s = "1 + 1"
输出：2
```

示例 2：

```
输入：s = " 2-1 + 2 "
输出：3
```

示例 3：

```
输入：s = "(1+(4+5+2)-3)+(6+8)"
输出：23
```


提示：

+ 1 <= s.length <= 3 * 105
+ s 由数字、'+'、'-'、'('、')'、和 ' ' 组成
+ s 表示一个有效的表达式



用了一个栈来存当前的符号要不要被翻转，栈里先push一个1，表示不用翻转，用了一个变量sign表示当前遇到的符号，sign和加减号有关，和栈顶元素也有关，如果栈顶元素为-1，表示当前需要翻转，遇到加号应该理解为减号，反之亦然。遇到括号时，需要向栈中push sign的值，因为sign是括号前的符号的实际含义。

```java
class Solution {
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        stack.push(1);
        int sign = 1;
        int res = 0;
        for(int i = 0; i < s.length(); ){
            if(s.charAt(i) == '+'){
                sign = stack.peek();
                i++;
            }
            else if(s.charAt(i) == '-'){
                sign = -stack.peek();
                i++;
            }
            else if(s.charAt(i) == '('){
                stack.push(sign);
                i++;
            }
            else if(s.charAt(i) == ')'){
                stack.pop();
                i++;
            }
            else if(s.charAt(i) == ' '){
                i++;
            }
            else{
                int sum = 0;
                while(i < s.length() && s.charAt(i) >= '0' && s.charAt(i) <= '9'){
                    sum = sum * 10 + s.charAt(i) - '0';
                    i++;
                }
                res = res + sign * sum;
            }
        }
        return res;
    }
}
```



#### 235 二叉搜索树的最近公共祖先（经典）

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

这道题用236的做法也能过，但是就没有利用到二叉搜索树的属性。

做过但是忘了。其实思路挺简单的。判断当前节点是大于两个、小于两个、大于一个小于一个，哪一种

如果是大于两个就往左子树遍历，小于两个就往右子树遍历，一大一小就是当前节点作为公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* ptr = root;
        while(ptr != NULL){
            if(ptr->val > p->val && ptr->val > q->val){
                ptr = ptr->left;
            }
            else if(ptr->val < p->val && ptr->val < q->val){
                ptr = ptr->right;
            }
            else{
                return ptr;
            }
        }
        return NULL;
    }
};
```



#### 236 二叉树的最近公共祖先（经典）

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

递归函数返回值挺特别的

如果左子树中包含一个结点，右子树也包含一个结点，说明公共祖先是当前结点。

如果左子树没有包含，说明公共祖先在右子树上。反之亦然。如果root为p或q，返回root

思路有点绕，其实可以理解为，p和q把自己的值往上传给它们的父结点，直到有一个点它的左右各有一个p或q

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null) return null;
        if(root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if(left != null && right != null) return root;
        if(left != null) return left;
        if(right != null) return right;
        return null;
    }
}
```



#### 239 滑动窗口最大值（单调双端队列）

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

参考了[五分钟学算法](https://mp.weixin.qq.com/s?__biz=MzUyNjQxNjYyMg==&mid=2247498838&idx=2&sn=c10692783ae0bee1312ffe1aa7d10339&chksm=fa0d93d7cd7a1ac1ceff349bfe2a67770bdcebca3bd8a13c5148e6aacab087dbee8154cbeea8&scene=126&sessionid=1616233063&key=4764af25be59e44a2a6d0a22e84b56ef4f29cd5450586618381eda3dcc1a89ac4bc5c790ef342db8ed6bb61acafba342aa10f2ddac5900b381931c725a65f0db410c00813cad0ed142edc3df638b4c586d9a073685ba6c1f02a63c1f9e7c3e36cc53a2217df5d370a5a423bc947a7107213ecd9792c609729ee34dc8950465d8&ascene=1&uin=Mzg0Njg0NzU2&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A%2BQKkwMXJve5HGlpk1VN8u0%3D&pass_ticket=R%2F0%2Fb2w4jP8VfyFlSRsubdTmnhgWNUHpUDtMa%2FGe957YH6%2BYbTc3O%2FLJjZZMGFnF&wx_header=0)里的图解写出下面的代码，这里用的是单调递减队列，**队列里存的是元素的下标，当队首的下标比窗口的左边界要小的时候，需要把队首移除**

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        int[] res = new int[len - k + 1];
        Deque<Integer> queue = new LinkedList<>();
        for(int i = 0; i < len; i++){
            while(!queue.isEmpty() && nums[queue.peekLast()] < nums[i]){
                queue.removeLast();
            }
            queue.addLast(i);
            if(i - queue.peek() >= k){
                queue.removeFirst();
            }
            if(i >= k - 1){
                res[i - k + 1] = nums[queue.peek()];
            }
        }
        return res;
    }
}
```

**注意**和剑指offer的59-1相同，但是代码复制过去却不行。

**原因** 剑指里输入的数组可能长度为0，需要增加一个判断，长度为零时返回[]



#### 278 第一个错误的版本（二分查找，简单题）

不写了



#### 283 移动零（原地修改数组）

给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意** ，必须在不复制数组的情况下原地对数组进行操作。

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int nextPos = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] != 0){
                nums[nextPos] = nums[i];
                ++nextPos;
            }
        }
        for(int i = nextPos; i < nums.length; i++){
            nums[i] = 0;
        }
    }
}
```



#### 287 寻找重复数（找链表中的环）

给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。

假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。

你设计的解决方案必须不修改数组 nums 且只用常量级 O(1) 的额外空间，O(n)的时间。

这道题限制特别多，完全没有思路。

https://leetcode-cn.com/problems/find-the-duplicate-number/solution/287xun-zhao-zhong-fu-shu-by-kirsche/

这个题解特别好，用图解释了怎么把这个数组转化为链表，这个问题就变成了链表里找环的问题，后面就好做了

142 题中慢指针走一步 slow = slow.next ==> 本题 slow = nums[slow]

142 题中快指针走两步 fast = fast.next.next ==> 本题 fast = nums[nums[fast]]

```java
class Solution {
    public int findDuplicate(int[] nums) {
        int fast = 1, slow = 0;
        while(fast != slow){
            fast = nums[nums[fast]];
            slow = nums[slow];
        }
        slow = nums[slow];
        fast = 0;
        while(fast != slow){
            fast = nums[fast];
            slow = nums[slow];
        }
        return slow;
    }
}
```



#### 295 数据流的中位数

中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

+ void addNum(int num) - 从数据流中添加一个整数到数据结构中。
+ double findMedian() - 返回目前所有元素的中位数。

示例：

```
addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
```

看了题解才想起来咋做，需要维护两个优先级队列，一个的peek是队列里的最大值（从大到小排），但存的是小的那半，另一个的peek是队列里的最小值（从小到大排），但存的是大的那半

让我比较混乱的是又要讨论放哪边，又要讨论奇数偶数，一开始就晕了。其实是只需要先和小的那半的最大值，queueMin.peek()比较，如果小于等于，就把数字放小的那半，如果大于，就放大的那半。先放了，然后再判断需不需要把小的那半移动一个过去大的那半，或者反过来

```java
class MedianFinder {
    Queue<Integer> queueMax, queueMin;
    boolean flag;// 标记总数的奇数还是偶数，偶数是true
    public MedianFinder() {
        queueMin = new PriorityQueue<Integer>(new Comparator<Integer>(){
            public int compare(Integer o1, Integer o2){
                return o2 - o1;
            }
        });
        queueMax = new PriorityQueue<Integer>();
        flag = true;
    }
    
    public void addNum(int num) {
        if(queueMin.isEmpty() || num <= queueMin.peek()){
            queueMin.add(num);
            if(queueMin.size() - queueMax.size() > 1){
                queueMax.add(queueMin.poll());
            }
        }
        else{
            queueMax.add(num);
            if(queueMax.size() > queueMin.size()){
                queueMin.add(queueMax.poll());
            }
        }
        flag = !flag;
    }
    
    public double findMedian() {
        if(!flag) return (double)queueMin.peek();
        else return (queueMin.peek() + queueMax.peek()) / 2.0;
    }
}
```



#### 297 二叉树的序列化与反序列化

我用的是层次遍历来做序列化与反序列化，两个函数的实现有点相似，都用到了队列，反序列化稍微难一点，这里简单介绍一下我的思路

先想到了用队列来存每一层的节点，出队的时候设置节点的左右孩子，并把左右孩子放进队列，和序列化不同的是这里的队列中没有null节点

另一个重点是用一个index来记录接下来的左右孩子是谁，这是最重要的，一开始我总觉得要用某个表达式计算出来，后来才意识到应该就是一个遍历数组的变量，数组中是有null的，当遍历到null时，表示当前节点的左/右孩子为null

~~仔细想想也不是非得要队列，再拿一个变量存接下来的父结点是谁也是可以的，遇到null就continue~~不行，因为数组里没有存TreeNode，所以还是需要一个变长的数据结构来存TreeNode，所以还是要一个队列

```java
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        List<String> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if(node == null) list.add("null");
            else {
                list.add(String.valueOf(node.val));
                queue.add(node.left);
                queue.add(node.right);
            }
        }
        return String.join(",", list); // ！！这里不能用‘-’，因为可能有负数，后面split会受影响
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] strs = data.split(",");
        if(strs.length == 1) return null;
        Queue<TreeNode> queue = new LinkedList<>(); // 非空的node才放进去
        TreeNode root = new TreeNode(Integer.parseInt(strs[0]));
        queue.add(root);
        for(int i = 1; i < strs.length;){
            TreeNode node = queue.poll();
            if(!strs[i].equals("null")){
                node.left = new TreeNode(Integer.parseInt(strs[i]));    
                queue.add(node.left);
            }
            i++;
            if(!strs[i].equals("null")){
                node.right = new TreeNode(Integer.parseInt(strs[i]));
                queue.add(node.right);
            }
            i++;    
        }
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));
```

下面这篇总结了所有解法

https://mp.weixin.qq.com/s/DVX2A1ha4xSecEXLxW_UsA

前序

```java
String SEP = ",";
String NULL = "#";

/* 主函数，将二叉树序列化为字符串 */
String serialize(TreeNode root) {
    StringBuilder sb = new StringBuilder();
    serialize(root, sb);
    return sb.toString();
}

/* 辅助函数，将二叉树存入 StringBuilder */
void serialize(TreeNode root, StringBuilder sb) {
    if (root == null) {
        sb.append(NULL).append(SEP);
        return;
    }

    /****** 前序遍历位置 ******/
    sb.append(root.val).append(SEP);
    /***********************/

    serialize(root.left, sb);
    serialize(root.right, sb);
}
/* 主函数，将字符串反序列化为二叉树结构 */
TreeNode deserialize(String data) {
    // 将字符串转化成列表
    LinkedList<String> nodes = new LinkedList<>();
    for (String s : data.split(SEP)) {
        nodes.addLast(s);
    }
    return deserialize(nodes);
}

/* 辅助函数，通过 nodes 列表构造二叉树 */
TreeNode deserialize(LinkedList<String> nodes) {
    // if (nodes.isEmpty()) return null; 
    // 这个没有也可以，不会进这个if的

    /****** 前序遍历位置 ******/
    // 列表最左侧就是根节点
    String first = nodes.removeFirst();
    if (first.equals(NULL)) return null;
    TreeNode root = new TreeNode(Integer.parseInt(first));
    /***********************/

    root.left = deserialize(nodes);
    root.right = deserialize(nodes);

    return root;
}
```

参考他的思路写了后序，虽然我的后序和他的有一点点不一样

```java
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root){
        StringBuilder strb = new StringBuilder();
        serialize(root, strb);
        return strb.toString();
    }
    void serialize(TreeNode root, StringBuilder strb){
        if(root == null){
            strb.append("NULL,");
            return;
        } 
        serialize(root.left, strb);
        serialize(root.right, strb);
        strb.append(root.val);
        strb.append(",");
    }
    int index;
    public TreeNode deserialize(String data){
        String[] nodes = data.split(",");
        index = nodes.length - 1;
        TreeNode root = deserialize(nodes);
        return root;
    }
    TreeNode deserialize(String[] nodes){
        // if(index < 0) return null;
        // 这行可以不要，不会进这个if
        
        if(nodes[index].equals("NULL")) {
            index--;
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(nodes[index]));
        index--;
        root.right = deserialize(nodes);
        root.left = deserialize(nodes);
        return root;
    }
}
```

后来我发现我这么做不行，因为不能有全局变量（不太懂为啥）



#### 300 最长递增子序列

给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

基于二分查找的做法，有点贪心的思想。需要维护一个递增的list，遍历数组时，如果当前元素大于list最大值，则直接append，否则二分查找，找到它应该插入的位置，替代该位置的值

```java
class Solution {
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if(len < 2) return len;
        List<Integer> list = new ArrayList<>();
        list.add(nums[0]);
        for(int i = 1; i < len; i++){
            if(nums[i] > list.get(list.size() - 1)){
                list.add(nums[i]);
            }
            else{
                int index = binarySearch(list, nums[i]);
                list.set(index, nums[i]);
            }
        }
        return list.size();
    }
    int binarySearch(List<Integer> list, int target){
        int low = 0, high = list.size() - 1;
        while(low <= high){
            int mid = low + (high - low) / 2;
            if(list.get(mid) == target) return mid;
            if(list.get(mid) > target) high = mid - 1;
            else low = mid + 1;
        }
        return low;
    }
}
```



#### 303 区域和检索-数组不可变（前缀和）

给定一个整数数组  nums，处理以下类型的多个查询:

计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right
实现 NumArray 类：

NumArray(int[] nums) 使用数组 nums 初始化对象
int sumRange(int i, int j) 返回数组 nums 中索引 left 和 right 之间的元素的 总和 ，包含 left 和 right 两点（也就是 nums[left] + nums[left + 1] + ... + nums[right] )


示例 1：

```
输入：
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
输出：
[null, 1, -1, -3]
```

```c++
class NumArray {
private:
    vector<int> preSum;

public:
    NumArray(vector<int>& nums) {
        preSum.push_back(0);
        for(int i = 0; i < nums.size(); i++){
            preSum.push_back(preSum[i] + nums[i]);
        }
    }
    
    int sumRange(int left, int right) {
        return preSum[right + 1] - preSum[left];
    }
};
```



#### 304 二维区域和检索-矩阵不可变（二维前缀和）

给定一个二维矩阵 matrix，以下类型的多个请求：

计算其子矩形范围内元素的总和，该子矩阵的 左上角 为 (row1, col1) ，右下角 为 (row2, col2) 。
实现 NumMatrix 类：

NumMatrix(int[][] matrix) 给定整数矩阵 matrix 进行初始化
int sumRegion(int row1, int col1, int row2, int col2) 返回 左上角 (row1, col1) 、右下角 (row2, col2) 所描述的子矩阵的元素 总和 。

其实还是有点难度的，关键要知道前缀和矩阵里preSum\[i]\[j]存的是从(0,0)到(i,j)这个矩阵里的数字的和

要求preSum[i]\[j]的时候，是用preSum[i]\[j - 1] + preSum[i - 1]\[j] - preSum[i - 1]\[j - 1] + nums[i - 1]\[j - 1]得到的，注意一定要加上nums里的值

求区间和的时候则是要用preSum[row2 + 1]\[col2 + 1] - preSum[row1]\[col2 + 1] - preSum[row2 + 1]\[col1] + preSum[row1]\[col1];

```c++
class NumMatrix {
private:
    vector<vector<int>> preSum;

public:
    NumMatrix(vector<vector<int>>& matrix) {
        int m = matrix.size();
        if(m > 0){
            int n = matrix[0].size();
            preSum.resize(m + 1, vector<int>(n + 1, 0));
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    preSum[i + 1][j + 1] = matrix[i][j] + preSum[i][j + 1] + preSum[i + 1][j] - preSum[i][j];
                }
            }
        }
        
    }
    
    int sumRegion(int row1, int col1, int row2, int col2) {
        return preSum[row2 + 1][col2 + 1] - preSum[row1][col2 + 1] - preSum[row2 + 1][col1] + preSum[row1][col1];
    }
};
```



#### 312 戳气球（区间DP，hard）

有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。

示例 1：

```
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
```

看了题解写的，题解写得非常好。

dp[i]\[j]表示开区间(i,j)所能获得的最大数量，这时候只需要关心最后一个被戳破的气球k。

dp[i]\[j] = max(dp[i]\[j], dp[i]\[k] + dp[k]\[j] + numsAppend[k] * numsAppend[i] * numsAppend[j]);

区间DP的做法是对span长度从小到大计算，外层循环是span长度，内层循环的起点位置。这道题span最短为3，因为是开区间。（为了避免边界问题，在前后都补了1）最内层循环就是最后一个戳破的气球的位置。

```c++
class Solution {
public:
    int maxCoins(vector<int>& nums) {
        vector<int> numsAppend;
        numsAppend.push_back(1);
        numsAppend.insert(numsAppend.end(), nums.begin(), nums.end());
        numsAppend.push_back(1);
        int len = numsAppend.size();
        vector<vector<int>> dp(len, vector<int>(len, 0));
        for(int span = 3; span <= len; span++){
            for(int i = 0; i <= len - span; i++){
                int j = i + span - 1;
                for(int k = i + 1; k < j; k++){
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + numsAppend[k] * numsAppend[i] * numsAppend[j]);
                }
            }
        }
        return dp[0][len - 1];
    }
};
```



#### 316 去除重复字母（单调栈，难想到）

给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

示例 1：

```
输入：s = "bcabc"
输出："abc"
```

示例 2：

```
输入：s = "cbacdcbc"
输出："acdb"
```


提示：

1 <= s.length <= 104
s 由小写英文字母组成

看了下面的题解做的

https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247486946&idx=1&sn=94804eb15be33428582544a1cd90da4d&scene=21#wechat_redirect

```java
class Solution {
    public String removeDuplicateLetters(String s) {
        int[] count = new int[256];
        char[] charArray = s.toCharArray();
        for(char c : charArray){
            count[c]++;
        }
        boolean[] instack = new boolean[256];
        Stack<Character> stack = new Stack<>();
        for(char c : charArray){
            count[c]--;
            if(instack[c]) continue;
            while(!stack.isEmpty() && stack.peek() > c){
                if(count[stack.peek()] == 0) break;
                instack[stack.pop()] = false;
            }
            stack.push(c);
            instack[c] = true;
        }
        StringBuilder strb = new StringBuilder();
        while(!stack.isEmpty()){
            strb.append(stack.pop());
        }
        return strb.reverse().toString();
    }
}
```



#### 322 零钱兑换（稍微有点难的完全背包，DP）

给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

我认为比较难搞的是要求最小值，而不存在的时候又要返回-1，最后决定先把不存在的时候赋值为最大值，最后返回的时候再改成-1，又出现了int越界的问题，解决的方式是再加一个判断保证它不会越界

优化前和优化后的代码：

```java
// class Solution {
//     public int coinChange(int[] coins, int amount) {
//         int len = coins.length;
//         int[][] dp = new int[len + 1][amount + 1];
//         for(int i = 0; i <= len; i++){
//             dp[i][0] = 0;
//         }
//         for(int j = 1; j <= amount; j++){
//             dp[0][j] = Integer.MAX_VALUE;
//         }
//         for(int i = 1; i <= len; i++){
//             for(int j = 1; j <= amount; j++){
//                 if(j >= coins[i - 1] && dp[i][j - coins[i - 1]] != Integer.MAX_VALUE){
//                     dp[i][j] = Math.min(dp[i - 1][j], 1 + dp[i][j - coins[i - 1]]);
//                 }
//                 else{
//                     dp[i][j] = dp[i - 1][j];
//                 }
//             }
//         }
//         return dp[len][amount] == Integer.MAX_VALUE ? -1 : dp[len][amount];
//     }
// }
class Solution {
    public int coinChange(int[] coins, int amount) {
        int len = coins.length;
        int[] dp = new int[amount + 1];

        for(int j = 1; j <= amount; j++){
            dp[j] = Integer.MAX_VALUE;
        }
        for(int i = 1; i <= len; i++){
            for(int j = 1; j <= amount; j++){
                if(j >= coins[i - 1] && dp[j - coins[i - 1]] != Integer.MAX_VALUE){
                    dp[j] = Math.min(dp[j], 1 + dp[j - coins[i - 1]]);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }
}
```



#### 337 打家劫舍三（树状）

记忆化+递归

```java
class Solution {
    Map<TreeNode, Integer> map = new HashMap<>();
    public int rob(TreeNode root) {
        if(root == null) return 0;
        if(map.containsKey(root)) return map.get(root);
        int robNow = root.val + (root.left == null ? 0 : rob(root.left.left) + rob(root.left.right))
                    + (root.right == null ? 0 : rob(root.right.left) + rob(root.right.right));
        int notRobNow = rob(root.left) + rob(root.right);
        
        int res = Math.max(robNow, notRobNow);
        map.put(root, res); // 记得这里要put
        return res;
    }
}
```

https://mp.weixin.qq.com/s/z44hk0MW14_mAQd7988mfw

```java
int rob(TreeNode root) {
    int[] res = dp(root);
    return Math.max(res[0], res[1]);
}

/* 返回一个大小为 2 的数组 arr
arr[0] 表示不抢 root 的话，得到的最大钱数
arr[1] 表示抢 root 的话，得到的最大钱数 */
int[] dp(TreeNode root) {
    if (root == null)
        return new int[]{0, 0};
    int[] left = dp(root.left);
    int[] right = dp(root.right);
    // 抢，下家就不能抢了
    int rob = root.val + left[0] + right[0];
    // 不抢，下家可抢可不抢，取决于收益大小
    int not_rob = Math.max(left[0], left[1])
                + Math.max(right[0], right[1]);

    return new int[]{not_rob, rob};
}
```



#### 353 贪吃蛇（双端队列）

请你设计一个 贪吃蛇游戏，该游戏将会在一个 屏幕尺寸 = 宽度 x 高度 的屏幕上运行。如果你不熟悉这个游戏，可以 点击这里 在线试玩。

起初时，蛇在左上角的 (0, 0) 位置，身体长度为 1 个单位。

你将会被给出一个数组形式的食物位置序列 food ，其中 food[i] = (ri, ci) 。当蛇吃到食物时，身子的长度会增加 1 个单位，得分也会 +1 。

食物不会同时出现，会按列表的顺序逐一显示在屏幕上。比方讲，第一个食物被蛇吃掉后，第二个食物才会出现。

当一个食物在屏幕上出现时，保证 不会 出现在被蛇身体占据的格子里。

如果蛇越界（与边界相撞）或者头与 移动后 的身体相撞（即，身长为 4 的蛇无法与自己相撞），游戏结束。

实现 SnakeGame 类：

SnakeGame(int width, int height, int[][] food) 初始化对象，屏幕大小为 height x width ，食物位置序列为 food
int move(String direction) 返回蛇在方向 direction 上移动后的得分。如果游戏结束，返回 -1 。

这道题其实就是模拟题，比较难的是怎么模拟蛇的移动，一开始想太复杂了，其实很简单

只要考虑蛇头和蛇尾就行，所以使用一个双端队列，如果没有碰到食物，就是在队尾去掉一个元素，在队头增加一个元素

每次移动可以分三种情况

+ 出界
+ 吃到食物，蛇尾不动，蛇头增加
+ 没吃到食物，蛇尾remove，蛇头add，要判断新增的蛇头是否已经在queue中，也就是是否和身体相撞

需要一些成员变量，其中queue里存的integer是row*width+column

```java
class SnakeGame {
    private int score, foodIndex, width, height;
    private int[][] food;
    private Deque<Integer> queue = new LinkedList<>();
    public SnakeGame(int width, int height, int[][] food) {
        this.food = food; 
        this.score = 0;
        this.width = width;
        this.height = height;
        this.foodIndex = 0;
        queue.add(0);
    }
    
    public int move(String direction) {
        int head = queue.peek();
        int row = head / width;
        int column = head % width;
        if(direction.equals("R")){
            column++;
        }
        else if(direction.equals("L")){
            column--;
        }
        else if(direction.equals("U")){
            row--;
        }
        else{
            row++;
        }
        if(row < 0 || row >= height || column < 0 || column >= width){
            return -1;
        }
        head = row * width + column;
        if(foodIndex < food.length && food[foodIndex][0] == row && food[foodIndex][1] == column){
            foodIndex++;
            queue.addFirst(head);
            return ++score;
        }

        queue.pollLast();
        if(queue.contains(head)){
            return -1;
        }
        queue.addFirst(head);
        return score;
    }
}

/**
 * Your SnakeGame object will be instantiated and called as such:
 * SnakeGame obj = new SnakeGame(width, height, food);
 * int param_1 = obj.move(direction);
 */
```



#### 354 俄罗斯套娃信封问题（最长单调递增子序列）

给你一个二维整数数组 envelopes ，其中 envelopes[i] = [wi, hi] ，表示第 i 个信封的宽度和高度。

当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里，如同俄罗斯套娃一样。

请计算 最多能有多少个 信封能组成一组“俄罗斯套娃”信封（即可以把一个信封放到另一个信封里面）。

注意：不允许旋转信封。


示例 1：

```
输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。
```

示例 2：

```
输入：envelopes = [[1,1],[1,1],[1,1]]
输出：1
```


提示：

+ 1 <= envelopes.length <= 5000
+ envelopes[i].length == 2
+ 1 <= wi, hi <= 104

这题真的难。题解说得挺好的，建议看[题解](https://leetcode-cn.com/problems/russian-doll-envelopes/solution/e-luo-si-tao-wa-xin-feng-wen-ti-by-leetc-wj68/)

> 必须要保证对于每一种 w 值，我们最多只能选择 1 个信封。
>
> 我们可以将 h 值作为排序的第二关键字进行降序排序，这样一来，对于每一种 w 值，其对应的信封在排序后的数组中是按照 h 值递减的顺序出现的，那么这些 h 值不可能组成长度超过 1 的严格递增的序列，这就从根本上杜绝了错误的出现。
>
> 关于最长严格递增子序列的做法分为两种，题解里也说得很清楚，一种是比较直白的线性DP，时间复杂度为O(N2)，另一种是基于二分查找的做法，有点贪心的思想。需要维护一个递增的list，遍历数组时，如果当前元素大于list最大值，则直接append，否则二分查找，找到它应该插入的位置，替代该位置的值（该位置的值比它大）

```java
class Solution {
    public int maxEnvelopes(int[][] envelopes) {
        int len = envelopes.length;
        if(len < 2) return len;
        Arrays.sort(envelopes, new Comparator<int[]>(){
            public int compare(int[] o1, int[] o2){
                if(o1[0] != o2[0]){
                    return o1[0] - o2[0];
                }
                else{
                    return o2[1] - o1[1];
                }
            }
        });
        List<Integer> list = new ArrayList<>();
        list.add(envelopes[0][1]);
        for(int i = 1; i < len; i++){
            if(envelopes[i][1] > list.get(list.size() - 1)){
                list.add(envelopes[i][1]);
            }
            else{
                int index = binarySearch(list, envelopes[i][1]);
                list.set(index, envelopes[i][1]);
            }
        }
        return list.size();
    }
    int binarySearch(List<Integer> list, int target){
        int low = 0, high = list.size() - 1;
        while(low <= high){
            int mid = low + (high - low) / 2;
            if(list.get(mid) == target) return mid;
            if(list.get(mid) > target) high = mid - 1;
            else low = mid + 1;
        }
        return low;
    }
}
```



#### 366 寻找二叉树的叶子节点（有点难的递归）

给你一棵二叉树，请按以下要求的顺序收集它的全部节点：

依次从左到右，每次收集并删除所有的叶子节点
重复如上过程直到整棵树为空


示例:

输入: [1,2,3,4,5]

          1
         / \
        2   3
       / \     
      4   5    

输出: [[4,5,3],[2],[1]]

这道题的思路是通过递归函数得到当前节点的高度，叶子节点高度为0，示例中的根节点高度为2。以高度为输出list的下标，将高度相同的节点放进同一组里

```java
class Solution {
    List<List<Integer>> list = new ArrayList<>();
    public List<List<Integer>> findLeaves(TreeNode root) {
        getHeight(root);
        return list;
    }
    public int getHeight(TreeNode root){
        if(root == null) return -1;
        int left = getHeight(root.left);
        int right = getHeight(root.right);
        int h = Math.max(left, right) + 1;
        if(list.size() <= h){
            list.add(new ArrayList<>());
        }
        list.get(h).add(root.val);
        return h;
    }
}
```



#### 370 区间加法（差分数组）

假设你有一个长度为 n 的数组，初始情况下所有的数字均为 0，你将会被给出 k 个更新的操作。

其中，每个操作会被表示为一个三元组：[startIndex, endIndex, inc]，你需要将子数组 A[startIndex ... endIndex]（包括 startIndex 和 endIndex）增加 inc。

请你返回 k 次操作后的数组。

```c++
class Solution {
    public int[] getModifiedArray(int length, int[][] updates) {
        int[] diff = new int[length];
        for(int i = 0; i < updates.length; i++){
            diff[updates[i][0]] += updates[i][2];
            if(updates[i][1] + 1 < length)
                diff[updates[i][1] + 1] -= updates[i][2];
        }
        int[] res = new int[length];
        res[0] = diff[0];
        for(int i = 1; i < length; i++){
            res[i] = res[i - 1] + diff[i];
        }
        return res;
    }
}
```



#### 394 字符串解码（栈，麻烦的题）

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

示例 1：

```
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

示例 2：

```
输入：s = "3[a2[c]]"
输出："accaccacc"
```

示例 3：

```
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

示例 4：

```
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```


提示：

1 <= s.length <= 30
s 由小写英文字母、数字和方括号 '[]' 组成
s 保证是一个 有效 的输入。
s 中所有整数的取值范围为 [1, 300] 

因为要考虑嵌套括号，所以采用的方法是遇到一个右括号的时候弹出栈中的字符，将其重复指定的次数后再放入栈中。例如3[a2[c]]变成3[acc]再变成accaccacc。最后把栈内字符弹出。为了编程方便，实际使用的是vector不是stack。

```c++
class Solution {
public:
    string getDigits(string& s, int& ptr){
        string res;
        while(isdigit(s[ptr])){
            res.push_back(s[ptr]);
            ++ptr;
        }
        return res;
    }

    string vecToString(vector<string>& vec){
        string res;
        for(string s : vec){
            res += s;
        }
        return res;
    }

    string decodeString(string s) {
        vector<string> stk;
        int ptr = 0;
        while(ptr < s.size()){
            if(isdigit(s[ptr])){
                stk.push_back(getDigits(s, ptr));
            }
            else if(isalpha(s[ptr]) || s[ptr] == '['){
                stk.push_back(string(1, s[ptr]));
                ++ptr;
            }
            else{
                vector<string> sub;
                while(stk.back() != "["){
                    sub.push_back(stk.back());
                    stk.pop_back();
                }
                reverse(sub.begin(), sub.end());
                stk.pop_back();
                int rep = stoi(stk.back());
                stk.pop_back();
                string str_rep;
                string str_sub = vecToString(sub);
                while(rep > 0){
                    str_rep += str_sub;
                    rep--;
                }
                stk.push_back(str_rep);
                ++ptr;
            }
        }
        return vecToString(stk);
    }
};
```



#### 410 分割数组的最大值（二分，和1011类似）

给定一个非负整数数组 nums 和一个整数 m ，你需要将这个数组分成 m 个非空的连续子数组。

设计一个算法使得这 m 个子数组各自和的最大值最小。

示例 1：

```
输入：nums = [7,2,5,10,8], m = 2
输出：18
解释：
一共有四种方法将 nums 分割为 2 个子数组。 
其中最好的方式是将其分为 [7,2,5] 和 [10,8] 。
因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。
```

示例 2：

```
输入：nums = [1,2,3,4,5], m = 2
输出：9
```

示例 3：

```
输入：nums = [1,4,4], m = 3
输出：4
```


提示：

+ 1 <= nums.length <= 1000
+ 0 <= nums[i] <= 106
+ 1 <= m <= min(50, nums.length)

看题目真的是一头雾水，看了[题解](https://mp.weixin.qq.com/s?__biz=MzAxODQxMDM0Mw==&mid=2247487594&idx=1&sn=a8785bd8952c2af3b19890aa7cabdedd&chksm=9bd7ee62aca067742c139cc7c2fa9d11dc72726108611f391d321cbfc25ccb8d65bc3a66762b&scene=21#wechat_redirect)还是有点懵，但是类比一下1011就好懂很多。把分成m个数组想成船要装m次，把子数组的和想成船的载重量，这道题就变成了，如果要在m次内装完货，船至少要有多少载重量。

代码就和1011非常相似了

```java
class Solution {
    public int splitArray(int[] nums, int m) {
        int left = 0, right = 0;
        for(int num : nums){
            left = Math.max(left, num);
            right += num;
        }
        while(left < right){
            int mid = left + (right - left) / 2;
            int split = needMinSplit(nums, mid);
            if(split == m){
                right = mid;
            }
            else if(split > m){
                left = mid + 1;
            }
            else{
                right = mid;
            }
        }
        return right;
    }
    int needMinSplit(int[] nums, int max){
        int res = 0;
        for(int i = 0; i < nums.length;){
            int cur = max;
            while(i < nums.length && cur >= nums[i]){
                cur -= nums[i];
                ++i;
            }
            res++;
        }
        return res;
    }
}
```



#### 415 字符串相加

给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和并同样以字符串形式返回。

你不能使用任何內建的用于处理大整数的库（比如 BigInteger）， 也不能直接将输入的字符串转换为整数形式。

```java
class Solution {
    public String addStrings(String num1, String num2) {
        StringBuilder res = new StringBuilder();
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        int sum = 0, add = 0, digit1 = 0, digit2 = 0;
        while(i >= 0 || j >= 0 || add > 0){
            digit1 = (i >= 0) ? (num1.charAt(i) - '0') : 0;
            digit2 = (j >= 0) ? (num2.charAt(j) - '0') : 0;
            sum = digit1 + digit2 + add; // 记得！
            add = sum / 10;
            res.append(sum % 10);
            i--; // 记得！
            j--;
        }
        return res.reverse().toString();
    }
}
```

+ 第一次死循环了，因为忘记i--，j--了



#### 416 分割等和子集（01背包）

给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

优化前和优化后的代码如下

```java
// class Solution {
//     public boolean canPartition(int[] nums) {
//         int sum = 0;
//         int len = nums.length;
//         for(int i = 0; i < len; i++){
//             sum += nums[i];
//         }
//         if(sum % 2 != 0) return false;
//         boolean[][] dp = new boolean[len + 1][sum / 2 + 1];
//         for(int i = 0; i <= len; i++){
//             dp[i][0] = true;
//         }
//         for(int i = 1; i <= len; i++){
//             for(int j = 1; j <= sum / 2; j++){
//                 dp[i][j] = dp[i - 1][j];
//                 if(j >= nums[i - 1]){
//                     dp[i][j] = dp[i][j] | dp[i - 1][j - nums[i - 1]];
//                 }
//             }
//         }
//         return dp[len][sum / 2];
//     }
// }
class Solution {
    public boolean canPartition(int[] nums) {
        int sum = 0;
        int len = nums.length;
        for(int i = 0; i < len; i++){
            sum += nums[i];
        }
        if(sum % 2 != 0) return false;
        boolean[] dp = new boolean[sum / 2 + 1];
        dp[0] = true;
        
        for(int i = 1; i <= len; i++){
            for(int j = sum / 2; j >= 0; j--){
                if(j >= nums[i - 1]){
                    dp[j] = dp[j] | dp[j - nums[i - 1]];
                }
                if(i == len) break;
            }
        }
        return dp[sum / 2];
    }
}
```



#### 438 找到字符串中所有字母异位词（滑动窗口）

给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。 

示例 1:

```
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
```

**提示:**

- `1 <= s.length, p.length <= 3 * 104`
- `s` 和 `p` 仅包含小写字母

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int[] countP = new int[26];
        int[] countS = new int[26];
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i < p.length(); i++){
            char c = p.charAt(i);
            countP[c - 'a']++;
        }
        int left = 0, right = 0;
        while(right < s.length()){
            char cur = s.charAt(right);
            countS[cur - 'a']++;
            if(right - left + 1 == p.length()){
                if(equal(countS, countP)){
                    res.add(left);
                }
                char del = s.charAt(left);
                countS[del - 'a']--;
                left++;
            }
            right++;
        }
        return res;
    }
    boolean equal(int[] nums1, int[] nums2){
        for(int i = 0; i < nums2.length; i++){
            if(nums1[i] != nums2[i]) return false;
        }
        return true;
    }
}
```

主要的点是用一个数组来记录每个字母出现的次数，如果用map开销会比较大



#### 440 字典序的第k小数字（类似第60题，剪枝）

给定整数 n 和 k，返回  [1, n] 中字典序第 k 小的数字。

示例 1:

```
输入: n = 13, k = 2
输出: 10
解释: 字典序的排列是 [1, 10, 11, 12, 13, 2, 3, 4, 5, 6, 7, 8, 9]，所以第二小的数字是 10。
```

示例 2:

```
输入: n = 1, k = 1
输出: 1
```


提示:

1 <= k <= n <= 109

看题解想了很久，发现和第60题很像，建议先掌握好第60题再做

本质是一样的，有一棵树，要找到先序遍历到的第k个节点。我们可以计算出以当前节点为根的树一共有多少节点，决定要不要进入这棵树（其实就是十叉树

```java
class Solution {
    public int findKthNumber(int n, int k) {
        // i表示前缀
        for(int i = 1; i <= n; ){
            long count = getCount(i, n);
            if(k == 1) return i; // 当前节点就是要找的
            if(k > count){ // 不进这棵树
                k -= count;
                i++;
                continue;
            }
            else{
                k--; // 进这棵树，减去根节点
                i *= 10;
            }
        }
        return -1;
    }
    // 记得用Long避免乘10之后溢出
    long getCount(int prefix, int upbound){
        long a = prefix, b = prefix + 1;
        long count = 0;
        for(; a <= upbound; b *= 10, a *= 10){
            count += (Math.min(upbound + 1, b) - a);
        }
        return count;
    }

}
```



#### 449 序列化和反序列化二叉搜索树（有点难，297变形）

因为是二叉搜索树，可以利用二叉搜索树的性质，所以不需要像297那样把null也加进去，就可以节省一些空间。

我一开始是做成前序遍历+中序遍历构造二叉树。结果超时了。。

看的题解的做法，比较类似98题的验证二叉树，是根据root的value来给左右子树限定值的范围

如果当前值在区间范围内，**就创建一个新的节点**，否则说明当前值不在这个区间，我们返回一个空节点。

这个有点难理解，可以画图

```java
public class Codec {

    // Encodes a tree to a single string.
    // 前序好做的
    public String serialize(TreeNode root) {
        StringBuilder strb = new StringBuilder();
        serialize(root, strb);
        return strb.toString();
    }
    void serialize(TreeNode root, StringBuilder strb){
        if(root == null) return;
        strb.append(String.valueOf(root.val));
        strb.append(",");
        serialize(root.left, strb);
        serialize(root.right, strb);
    }
    
    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        // 注意这个要特判空字符串
        if(data.length() == 0) return null;
        String[] strs = data.split(",");
        LinkedList<Integer> list = new LinkedList<>();
        for(int i = 0; i < strs.length; i++){
            list.add(Integer.parseInt(strs[i]));
        }
        return deserialize(list, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }
    TreeNode deserialize(LinkedList<Integer> list, int min, int max){
        if(list.isEmpty()) return null;
        // 当这个点的值不满足要求时，说明它不在这棵子树，它会被用在另一边
        if(list.getFirst() >= max || list.getFirst() <= min) return null;
        TreeNode root = new TreeNode(list.removeFirst());
        root.left = deserialize(list, min, root.val);
        root.right = deserialize(list, root.val, max);
        return root;
    }
}
```



#### 450 删除二叉搜索树中的节点（掌握模板）

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。

看了官方题解做的，一开始真的毫无思路

首先要确定找到要删的结点之后怎么删，这是最重要的！

+ 要拿它的前驱或者后继节点来替代它
+ 要怎么替代它？直接替吗？
+ 不行，这是这道题让我感觉到最绕的。联系链表里删除节点，1->2->3要删掉2的话，不能说node2=node3这样去删，一般我们用的是node1.next=node3，或者像这道题就是node2.val=node3.val, 然后去删除node3

现在总结一下。

+ 首先需要有两个函数用来求一个结点的前驱和后继
+ 当我们找到要删除的结点时，如果是叶子节点，直接赋值为null，返回；如果有右子树，则让当前节点赋值为后继节点的值，再去右子树里删掉这个后继节点；如果没有右子树，则让当前节点赋值为前驱节点的值，再去左子树里删掉这个前驱节点
+ 需要注意的是，如果是去左子树或者右子树里删除，删除完之后会返回新的子树的根，需要把这个返回值赋给当前节点的左或右孩子，这一步也非常重要，而且容易漏

```java
class Solution {
    TreeNode successor(TreeNode root) {
        root = root.right;
        while(root.left != null) root = root.left;
        return root;
    }
    TreeNode predecessor(TreeNode root) {
        root = root.left;
        while(root.right != null) root = root.right;
        return root;
    }
    public TreeNode deleteNode(TreeNode root, int key) {
        if(root == null) return null;
        if(root.val > key) root.left = deleteNode(root.left, key);
        else if(root.val < key) root.right = deleteNode(root.right, key);
        else{
            if(root.left == null && root.right == null) root = null;
            else if(root.right != null){
                root.val = successor(root).val;
                root.right = deleteNode(root.right, root.val);
            }
            else {
                root.val = predecessor(root).val;
                root.left = deleteNode(root.left, root.val);
            }
        }
        return root;
    }
}
```



#### 468 验证IP地址

编写一个函数来验证输入的字符串是否是有效的 IPv4 或 IPv6 地址。

如果是有效的 IPv4 地址，返回 "IPv4" ；
如果是有效的 IPv6 地址，返回 "IPv6" ；
如果不是上述类型的 IP 地址，返回 "Neither" 。
IPv4 地址由十进制数和点来表示，每个地址包含 4 个十进制数，其范围为 0 - 255， 用(".")分割。比如，172.16.254.1；

同时，IPv4 地址内的数不会以 0 开头。比如，地址 172.16.254.01 是不合法的。

IPv6 地址由 8 组 16 进制的数字来表示，每组表示 16 比特。这些组数字通过 (":")分割。比如,  2001:0db8:85a3:0000:0000:8a2e:0370:7334 是一个有效的地址。而且，我们可以加入一些以 0 开头的数字，字母可以使用大写，也可以是小写。所以， 2001:db8:85a3:0:0:8A2E:0370:7334 也是一个有效的 IPv6 address地址 (即，忽略 0 开头，忽略大小写)。

然而，我们不能因为某个组的值为 0，而使用一个空的组，以至于出现 (::) 的情况。 比如， 2001:0db8:85a3::8A2E:0370:7334 是无效的 IPv6 地址。

同时，在 IPv6 地址中，多余的 0 也是不被允许的。比如， 02001:0db8:85a3:0000:0000:8a2e:0370:7334 是无效的。

简直就是疯狂试错的一道题

```java
class Solution {
    public String validIPAddress(String queryIP) {
        if(queryIP.indexOf('.') != -1){ // 可以用queryIP.contains(".")
            if(queryIP.endsWith(".")) return "Neither"; // 结尾的不会被split划分进去，所以必须特判
            String[] ipv4 = queryIP.split("\\."); // split得先转义
            if(ipv4.length != 4) return "Neither";
            for(int i = 0; i < 4; i++){
                if(!isValidIPv4(ipv4[i])) return "Neither"; 
            }
            return "IPv4";
        }
        if(queryIP.indexOf(':') != -1){
            if(queryIP.endsWith(":")) return "Neither"; // 结尾的不会被split划分进去，所以必须特判
            String[] ipv6 = queryIP.split(":"); // .split("\\:")也可以
            if(ipv6.length != 8) return "Neither";
            for(int i = 0; i < 8; i++){
                if(!isValidIPv6(ipv6[i])) return "Neither";
            }
            return "IPv6";
        }
        return "Neither";
    }
    boolean isValidIPv4(String str){
        if(str.length() == 0 || str.length() > 3) return false; // 长度必须在1到3之间！
        if(str.startsWith("0") && str.length() > 1) return false; // 开头是0且不止一位才是错的，如果只有一位0就可以
        int sum = 0;
        for(int i = 0; i < str.length(); i++){
            if(str.charAt(i) > '9' || str.charAt(i) < '0') return false;
            sum = sum * 10 + str.charAt(i) - '0';
        }
        if(sum > 255) return false;
        return true;
    }
    boolean isValidIPv6(String str){
        if(str.length() == 0 || str.length() > 4) return false; // 长度必须在1到4之间！
        for(int i = 0; i < str.length(); i++){
            char c = str.charAt(i);
            if((c <= '9' && c >= '0') || (c <= 'f' && c >= 'a') || (c <= 'F' && c >= 'A')){ // 16进制是a到f不是a到e
                continue;
            }
            else return false;
        }
        return true;
    }
}
```

看了题解，最简单的是调库，这里就不放代码了

另一种做法是正则，正则可以避免非常多的if...else，代码会简洁很多，推荐

```java
import java.util.regex.Pattern;
class Solution {
  String chunkIPv4 = "([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])";
  Pattern pattenIPv4 =
          Pattern.compile("^(" + chunkIPv4 + "\\.){3}" + chunkIPv4 + "$");

  String chunkIPv6 = "([0-9a-fA-F]{1,4})";
  Pattern pattenIPv6 =
          Pattern.compile("^(" + chunkIPv6 + "\\:){7}" + chunkIPv6 + "$");

  public String validIPAddress(String IP) {
    if (IP.contains(".")) {
      return (pattenIPv4.matcher(IP).matches()) ? "IPv4" : "Neither";
    }
    else if (IP.contains(":")) {
      return (pattenIPv6.matcher(IP).matches()) ? "IPv6" : "Neither";
    }
    return "Neither";
  }
}
```



#### 484 寻找排列（找规律，贪心）

由范围 [1,n] 内所有整数组成的 n 个整数的排列 perm 可以表示为长度为 n - 1 的字符串 s ，其中:

如果 perm[i] < perm[i + 1] ，那么 s[i] == ' i '
如果 perm[i] > perm[i + 1] ，那么 s[i] == 'D' 。
给定一个字符串 s ，重构字典序上最小的排列 perm 并返回它。

示例 1：

```
输入： s = "I"
输出： [1,2]
解释： [1,2] 是唯一合法的可以生成秘密签名 "I" 的特定串，数字 1 和 2 构成递增关系。
```

示例 2：

```
输入： s = "DI"
输出： [2,1,3]
解释： [2,1,3] 和 [3,1,2] 可以生成秘密签名 "DI"，
但是由于我们要找字典序最小的排列，因此你需要输出 [2,1,3]。
```


提示：

1 <= s.length <= 105
s[i] 只会包含字符 'D' 和 'I'。

多举几个例子就可以找到规律啦，一开始先假设全是I，答案就是顺序的12345...如果遇到D，就需要reverse，比如12变成21，如果是两个D，就是123变成321，所以说遇到几个D，就翻转（几加一）个数字

```java
class Solution {
    public int[] findPermutation(String s) {
        int[] res = new int[s.length() + 1];
        for(int i = 0; i < res.length; i++){
            res[i] = i + 1;
        }
        
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == 'D'){
                int j = i;
                while(j < s.length() && s.charAt(j) == 'D'){
                    j++;
                }
                int left = i;
                int right = j;
                int tmp = 0;
                while(left < right){
                    tmp = res[left];
                    res[left] = res[right];
                    res[right] = tmp;
                    left++;
                    right--;
                }
                i = j;
            }
            
        }
        return res;
    }
}
```



#### 一些见解

+ 如果有一个二维数组，且只能向右向下走，很大概率就是二维DP，dp\[i][j]表示的是到(i,j)这个点的结果
+ 字符串编辑距离，正则匹配，最长公共子序列，二维DP
+ 在树/图中求最短距离，BFS
+ 排列组合，回溯