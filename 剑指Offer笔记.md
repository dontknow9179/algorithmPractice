## 剑指Offer笔记

#### 05 替换空格

请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。

```java
class Solution {
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder();
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) == ' '){
                res.append("%20");
            }
            else{
                res.append(s.charAt(i));
            }
        }
        return res.toString();
    }
}
```

时间复杂度O(N)，空间复杂度O(N)

不用string builder用string的话会慢很多



#### 06 从尾到头打印链表

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public int[] reversePrint(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        ListNode iter = head;
        while(iter != null){
            stack.push(iter.val);
            iter = iter.next;
        }
        int size = stack.size();
        int[] res = new int[size];
        for(int i = 0; i < size; i++){
            res[i] = stack.pop();
        }
        return res;
    }
}
```

第一次用了`Collections.reverse`方法来颠倒，结果速度挺慢

后来看了题解，利用stack先进后出的特点，还是挺慢的。。。



#### 07 重建二叉树

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    HashMap<Integer, Integer> map;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int len = preorder.length;
        map = new HashMap<>();
        for(int i = 0; i < len; i++){
            map.put(inorder[i], i);
        }
        if(len == 0) return null;
        TreeNode root = buildTree(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
        return root;
    }
    public TreeNode buildTree(int[] preorder, int[] inorder, int ps, int pe, int is, int ie){

        // if(ps == pe){
        //     TreeNode root = new TreeNode(preorder[ps]);
        //     root.left = null;
        //     root.right = null;
        //     return root;
        // }
        // 这是一开始的做法，会报out of bounds的错
        // 例子：[1,2] [2,1]
        
        if(pe < ps){            
            return null;
        }
        else{
            TreeNode root = new TreeNode(preorder[ps]);
            int index = map.get(preorder[ps]);
            root.left = buildTree(preorder, inorder, ps+1, ps+index-is, is, index-1);
            root.right = buildTree(preorder, inorder, ps+index+1-is, pe, index+1, ie);
            return root;
        }    
    }
}
```

因为没考虑到可能会没有左/右子树，导致越界

只能说把这个情况背下来吧

**在数组里查找一个值的位置可以用字典建立（值-位置）映射，只需要遍历一遍，否则将遍历n遍**



#### 09 用两个栈实现队列

```java
class CQueue {
    Stack<Integer> stack1;
    Stack<Integer> stack2;
    public CQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    
    public void appendTail(int value) {
        
        stack1.push(value);
    }
    
    public int deleteHead() {
        if(stack2.isEmpty() && stack1.isEmpty()){
            return -1;
        }
        else if(stack2.isEmpty()){
            // int tmp = 0;
            while(!stack1.isEmpty()){
                // tmp = stack1.pop();
                // 把上面这个注释掉的代码改成下面这样，去掉中间的变量赋值可以减少执行用时
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
}

/**
 * Your CQueue object will be instantiated and called as such:
 * CQueue obj = new CQueue();
 * obj.appendTail(value);
 * int param_2 = obj.deleteHead();
 */
```



#### 11 旋转数组的最小数字（我觉得很难）

```
输入：[3,4,5,1,2]
输出：1
```

```
输入：[1,1,3]
输出：1
```

```
输入：[2,2,2,0,1]
输出：0
```

```java
class Solution {
    public int minArray(int[] numbers) {
        int len = numbers.length;
        
        int low = 0;
        int high = len - 1;
        int mid = 0;
        while(low < high){
            mid = low + (high - low) / 2;
            if(numbers[mid] < numbers[high]){
                high = mid;
            }
            else if(numbers[mid] > numbers[high]){
                low = mid + 1; // 注意这里要加一，否则会死循环
            }
            else{
                high--; //这里也是一个难点
            }
            // if(numbers[mid] < numbers[low]){
            //     high = mid - 1;
            // }
            // else if(numbers[mid] > numbers[low]){
            //     low = mid;
            // }
            // else{
            //     low++;
            // }
            // 注释掉的这种不行！
        }
        return numbers[high]; // return numbers[low]也可以
    }
}
```

这题虽然标为简单但是我做了很久，踩了很多坑，姑且背下来吧



#### 12 矩阵中的路径

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。

```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```

```
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

```java
class Solution {
    public boolean exist(char[][] board, String word) {
        int high = board.length;
        int wide = board[0].length;
        for(int i = 0; i < high; i++){
            for(int j = 0; j < wide; j++){
                if(exist(board, word, 0, i, j)) return true;
            }
        }
        return false;
    }
    int[][] directions = {{0, 1},{0, -1},{1, 0},{-1, 0}};
    public boolean exist(char[][] board, String word, int index, int x, int y){
        if(x >= board.length || y >= board[0].length || x < 0 || y < 0) return false;
        if(board[x][y] != word.charAt(index)) return false;
        if(index == word.length() - 1) return true;
        // 这三行重要！
        board[x][y] = '\0';
        for(int i = 0; i < 4; i++){
            if(exist(board, word, index+1, x+directions[i][0], y+directions[i][1])) 
                return true;
        }
        board[x][y] = word.charAt(index);
        return false;
    }

    // public boolean exist(char[][] board, String word) {
    //     int high = board.length;
    //     int wide = board[0].length;
    //     boolean[][] visited = new boolean[high][wide];
    //     for(int i = 0; i < high; i++){
    //         for(int j = 0; j < wide; j++){
    //             if(exist(board, visited, word, 0, i, j)) return true;
    //         }
    //     }
    //     return false;
    // }
    // int[][] directions = {{0, 1},{0, -1},{1, 0},{-1, 0}};
    // public boolean exist(char[][] board, boolean[][] visited, String word, int index, int x, int y){
    //     if(x >= board.length || y >= board[0].length || x < 0 || y < 0) return false;
    //     if(board[x][y] != word.charAt(index) || visited[x][y]) return false;
    //     if(index == word.length() - 1) return true;
    //     visited[x][y] = true;
    //     for(int i = 0; i < 4; i++){
    //         if(exist(board, visited, word, index+1, x+directions[i][0], y+directions[i][1])) 
    //             return true;
    //     }
    //     visited[x][y] = false;
    //     return false;
    // }
}
```

注释掉的做法是用了一个boolean\[][]，会消耗多一点的内存，没注释的做法是直接修改原来的char\[][]

要注意判断数组有没有越界，包括字符串和board数组，还要记得在完全匹配时返回true



#### 13 机器人运动范围

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

```java
class Solution {
    int m, n;
    public int movingCount(int m, int n, int k) {
        boolean[][] forbidden = new boolean[m][n];
        boolean[][] visited = new boolean[m][n];
        this.m = m;
        this.n = n;
        return movingCount(forbidden, visited, k, 0, 0);
    }
    int[][] directions = {{1,0},{0,1}};
    public int movingCount(boolean[][] forbidden, boolean[][] visited, int k, int x, int y){
        if(x >= this.m || y >= this.n || forbidden[x][y] || visited[x][y]){
            return 0;
        }

        if(compute(x) + compute(y) > k){
            forbidden[x][y] = true;
            return 0;
        }
        
        else{
            visited[x][y] = true;
            int count = 0;
            for(int i = 0; i < 2; i++){
                count += movingCount(forbidden, visited, k, x + directions[i][0], y + directions[i][1]);
            }
            return count + 1;
        }
        
        
    }
    public int compute(int num){
        int sum = 0;
        while(num != 0){
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }
}
```

我用的是深搜+递归，也可以用广搜+队列

**特点**：只需要考虑**向右和向下**两个方向，因为是从(0,0)开始走的，每个格子的左边和上面都会被覆盖到。大家只要都负责好右和下就行。



#### 14 剪绳子 

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

```java
class Solution {
    public int cuttingRope(int n) {
        if(n == 2) return 1;
        if(n == 3) return 2;
        long res = 1;
        while(n > 4){
            res = res * 3 % 1000000007;
            n -= 3;
        }
        return (int)(res * n % 1000000007);

        // if(n % 3 == 1){
        //     return (int)(Math.pow(3, n / 3 - 1) * 4 % 1000000007);
        // }
        // else if(n % 3 == 2){
        //     return (int)(Math.pow(3, n / 3) * 2 % 1000000007);
        // }
        // else{
        //     return (int)(Math.pow(3, n / 3) % 1000000007);
        // }
    }
}
```

**注意**：被注释掉的这种做法算出来结果不一样，原因还不太确定，可能是浮点数类型转换时有精度损失



#### 15 二进制中1的个数

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int count = 0;
        while(n != 0){
            n = n & (n - 1);
            count++;
        }
        return count;
    }
}
```

**记住**：二进制数和它减一按位与会消去它的一个1



#### 16 数值的整数次方

-100.0<x<100.0

-2147483648<=n<=2147483647

```java
// class Solution {
//     public double myPow(double x, int n) {
//         long exp;
//         double base;
//         if(n == 0) return 1;
//         if(n == -2147483648){
//             return pow(1/x, 2147483647) * (1/x);
//         }
//         if(n < 0){
//             exp = (-n);
//             base = 1 / x;
//         }
//         else{
//             exp = n;
//             base = x;
//         }
//         return pow(base, exp);
//     }
//     public double pow(double base, long exp){
//         if(exp == 0) return 1;
//         if(exp == 1) return base;
//         if(exp % 2 == 1){
//             return Math.pow(pow(base, exp / 2), 2) * base;
//         }
//         else{
//             return Math.pow(pow(base, exp / 2), 2);
//         }
//     }
// }


class Solution {
    public double myPow(double x, int n) {
        if(n == 0) return 1;
        long b = n;
        double res = 1.0;
        if(b < 0) {
            x = 1 / x;
            b = -b;
        }
        while(b > 0) {
            if((b & 1) == 1) res *= x;
            x *= x;
            b >>= 1;
        }
        return res;
    }
}
```

注释掉的是我写的使用递归函数二分思想做的

没注释掉的用的是**快速幂**和位运算做的，时间差别不大，代码更简洁

**注意：**-2147483648变成正的时候会超出范围，所以先把-2147483648放进long再转为正的，我一开始用`long exp = -n`这样是不行的

##### 快速幂

```java
int fast(int a, int b, int mod){
	int answer = 1;
    while(b != 0){         //不断将b转换为二进制数
        if(b % 2 == 1){    //若当前位为1，累乘a的2^k次幂
            answer *= a;
            answer %= mod;
        }
        b /= 2;
        a *= a;
        a %= mod;
    }
    return answer;
}
```



#### 17 打印从1到最大的n位数

```java
class Solution {
    public int[] printNumbers(int n) {
        int size = (int)Math.pow(10, n) - 1;
        int[] res = new int[size];
        for(int i = 0; i < size; i++){
            res[i] = i + 1;
        }
        return res;
    }
}
```

因为题目的返回值是int数组所以要简单很多，如果要考虑大数的话需要利用字符串+dfs遍历，相当于每一步的选择为0-9



#### 18 删除链表的节点

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点

```
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        // ListNode pre = head;
        // ListNode iter = head.next;
        // if(head.val == val){
        //     //head.next = null; 
        //     return iter;
        // }
        
        // while(iter != null){
        //     if(iter.val == val){              
        //         pre.next = iter.next;
        //         //iter.next = null;
        //         break;
        //     }
        //     pre = iter;
        //     iter = iter.next;
        // }
        // return head;
        if(head.val == val) return head.next;
        ListNode iter = head;
        while(iter.next != null){
            if(iter.next.val == val){
                iter.next = iter.next.next;
                break;
            }
            iter = iter.next;
        }
        return head;
    }
}
```

注释掉的是使用了双指针的做法，没注释掉的没有定义双指针

**注意**：第一次做的时候超出时间限制了想好久没想出来，后来才发现是循环里面没有将指针指向下一个节点



#### 19 正则表达式匹配

请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

- `s` 可能为空，且只包含从 `a-z` 的小写字母。
- `p` 可能为空，且只包含从 `a-z` 的小写字母以及字符 `.` 和 `*`，无连续的 `'*'`

> 我们不妨换个角度考虑这个问题：字母 + 星号的组合在匹配的过程中，本质上只会有两种情况：
>
> 匹配 s 末尾的一个字符，将该字符扔掉，而该组合还可以继续进行匹配；
>
> 不匹配字符，将该组合扔掉，不再进行匹配。
>
> 链接：https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/solution/zheng-ze-biao-da-shi-pi-pei-by-leetcode-s3jgn/

根据官方题解写的如下代码

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int s_len = s.length();
        int p_len = p.length();
        String s_ = "_" + s;
        String p_ = "_" + p;
        if(s_len == 0 && p_len == 0) return true;

        boolean[][] match = new boolean[s_len + 1][p_len + 1];
        match[0][0] = true;
        for(int j = 1; j <= p_len; j++){
            if(p_.charAt(j) != '*') match[0][j] = false;
            else{
                match[0][j] = match[0][j - 2];
            }
        }
        for(int i = 1; i <= s_len; i++){
            for(int j = 1; j <= p_len; j++){
                if(p_.charAt(j) != '*'){
                    if(match(s_.charAt(i), p_.charAt(j))){
                        match[i][j] = match[i - 1][j - 1];
                    }
                    else{
                        match[i][j] = false;
                    }
                }
                else{
                    if(match(s_.charAt(i), p_.charAt(j - 1))){
                        match[i][j] = match[i - 1][j] || match[i][j - 2];
                    }
                    else{
                        match[i][j] = match[i][j - 2];
                    }
                }
                
            }
        }
        return match[s_len][p_len];
    }
    public boolean match(char i, char j){
        if(i == j || j == '.') return true;
        else return false;
    }
}
```

**bug**:

+ 数组越界
+ "" 和 "a*" 应该返回true，解决办法是在字符前加了一个'_'
+ i和j并不是都从1开始，i=0时不一定为false

其实可以不用加'_'但是加了会更好懂点



#### 20 表示数值的字符串（确定有限自动状态机）

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是

参考了官方题解https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/solution/biao-shi-shu-zhi-de-zi-fu-chuan-by-leetcode-soluti/

```java
class Solution {
    public boolean isNumber(String s) {
        Map<Integer, Map<Character, Integer>> transfer = new HashMap<Integer, Map<Character, Integer>>();
        Map<Character, Integer> map1 = new HashMap<>(){{
            put(' ', 1);
            put('d', 3);
            put('s', 2);
            put('.', 6);
        }};
        Map<Character, Integer> map2 = new HashMap<>(){{
            put('d', 3);
            put('.', 6);
        }};
        Map<Character, Integer> map3 = new HashMap<>(){{
            put('d', 3);
            put('.', 4);
            put('e', 7);
            put(' ', 10);
        }};
        Map<Character, Integer> map4 = new HashMap<>(){{
            put('d', 5);
            put('e', 7);
            put(' ', 10);
        }};
        Map<Character, Integer> map5 = new HashMap<>(){{
            put('d', 5);
            put('e', 7);
            put(' ', 10);
        }};
        Map<Character, Integer> map6 = new HashMap<>(){{
            put('d', 5);
        }};
        Map<Character, Integer> map7 = new HashMap<>(){{
            put('d', 9);
            put('s', 8);
        }};
        Map<Character, Integer> map8 = new HashMap<>(){{
            put('d', 9);
        }};
        Map<Character, Integer> map9 = new HashMap<>(){{
            put('d', 9);
            put(' ', 10);
        }};
        Map<Character, Integer> map10 = new HashMap<>(){{
            put(' ', 10);
        }};
        transfer.put(1, map1);
        transfer.put(2, map2);
        transfer.put(3, map3);
        transfer.put(4, map4);
        transfer.put(5, map5);
        transfer.put(6, map6);
        transfer.put(7, map7);
        transfer.put(8, map8);
        transfer.put(9, map9);
        transfer.put(10, map10);

        Integer state = 1; 
        Character curr;
        for(int i = 0; i < s.length(); i++){
            if(s.charAt(i) >= '0' && s.charAt(i) <= '9') curr = 'd';
            else if(s.charAt(i) == '+' || s.charAt(i) == '-') curr = 's';
            else if(s.charAt(i) == 'e' || s.charAt(i) == 'E') curr = 'e';
            else curr = s.charAt(i);
            if(!transfer.get(state).containsKey(curr)) return false;
            else{
                state = transfer.get(state).get(curr);
            }
        }
        return state == 3 || state == 4 || state == 5 || state == 9 || state == 10;
    }
}
```



#### 21 调整数组让奇数位于偶数前面

https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/solution/ti-jie-shou-wei-shuang-zhi-zhen-kuai-man-shuang-zh/

**法一**

看了题解的第二种思路写出来的

```java
class Solution {
    public int[] exchange(int[] nums) {
        int fast = 0, slow = 0;
        int tmp = 0;
        while(fast < nums.length){
            if(nums[fast] % 2 == 1){
                tmp = nums[fast];
                nums[fast] = nums[slow];
                nums[slow] = tmp;
                slow++;
            }       
            fast++;        
        }
        return nums;
    }
}
```

用了快慢两个指针，快指针每次都走一步，用于搜索奇数位置，慢指针指向的是奇数应该放的位置，所以慢指针不主动走，只在交换位置之后向前挪一格。

**法二**

首尾双指针，类似快排中的partition

左指针和右指针，左指针一直右移直到偶数，右指针一直左移直到奇数，交换一下，直到两者重合

感觉很好理解但是很容易写错，很容易发生数组越界的问题

```java
class Solution {
    public int[] exchange(int[] nums) {
        int left = 0, right = nums.length - 1;
        int tmp = 0;
        while(left < right){
            if(nums[left] % 2 == 1){
                left++;
                continue;
            }
            if(nums[right] % 2 == 0){
                right--;
                continue;
            }
            tmp = nums[left];
            nums[left] = nums[right];
            nums[right] = tmp;
            left++;
            right--;
        } 
        return nums;
    }
}
```



#### 22 链表中倒数第k个节点

```
给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
```

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode res = head;
        ListNode iter = head;
        while(k > 0){
            iter = iter.next;
            k--;
        }
        while(iter != null){
            res = res.next;
            iter = iter.next;
        }
        return res;
    }
}
```

双指针，差点忘记这个做法了，想了一下想起来了，只需一次遍历即可



#### 23 反转链表

```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

太久没做一下子忘记了，看了题解还是没什么印象。https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/solution/fan-zhuan-lian-biao-by-leetcode-solution-jvs5/

用了**三个指针**，一个指向前一个指向后一个指向当前，重要的两点是：pre初始为null，循环结束条件是`curr==null`

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode curr = head;
        ListNode pre = null;
        ListNode next;
        while(curr != null){
            next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }
}
```

一个奇怪的语法问题，前面定义三个ListNode不能用逗号分开写在同一行

第二种是**递归**，我虽然知道可以递归但是也忘记具体要怎么做了

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode newHead = reverseList(head.next);
        head.next.next = head;//重点
        head.next = null;
        return newHead;
    }
}
```

递归函数的返回值是反转后的链表头，递归结束时的链表尾的next是null，但是'head'的next指向的是递归结束时的链表尾，所以可以用`head.next.next = head`来把它变成新的链表尾



#### 25 合并两个排好序的链表

```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```

类似于归并排序，需要添加一个伪头节点否则第一轮迭代不好做

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode fake = new ListNode(0);
        ListNode iter = fake;
        while(l1 != null && l2 != null){
            if(l1.val <= l2.val){
                iter.next = l1;
                l1 = l1.next;
            }
            else{
                iter.next = l2;
                l2 = l2.next;
            }
            iter = iter.next;
        }
        if(l2 != null) iter.next = l2;
        if(l1 != null) iter.next = l1;
        return fake.next;//返回值是这个
    }
}
```



#### 26 树的子结构

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:

```
给定的树 A:
     3
    / \
   4   5
  / \
 1   2
给定的树 B：
   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。
```

其实不算难，比较难的一个地方是这里用了两个递归函数，一开始没想到其实要用两个递归

参考了题解 https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/solution/shi-shang-luo-ji-zui-qing-chu-zui-rong-y-2q93/ 的思路，和前一个题解的代码，自己写了下面这个

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        boolean res = false;
        if(A == null || B == null) return false;
        if(A.val == B.val) res = isSubTree(A, B);
        res = res || isSubStructure(A.left, B) || isSubStructure(A.right, B);
        return res; 
    }
    public boolean isSubTree(TreeNode A, TreeNode B){
        if(B == null) return true;
        if(B != null && A == null) return false;
        return A.val == B.val && isSubTree(A.left, B.left) && isSubTree(A.right, B.right);
    }
}
```

第一个函数虽然返回值是最终结果，但其实它是在找和B的根节点值相同的A节点，第二个函数是找到相同值的根节点后递归地进行比较，判断A是否包含了B。需要注意的是节点为null时应该返回true还是false



#### 27 二叉树的镜像

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if(root == null) return null;
        TreeNode leftRoot = mirrorTree(root.right);
        TreeNode rightRoot = mirrorTree(root.left);
        root.left = leftRoot;
        root.right = rightRoot;
        return root;
    }
}
```

时间复杂度O(N) 空间复杂度O(N)



#### 28 对称的二叉树

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

```
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
    1
   / \
  2   2
   \   \
   3    3
```

一开始没理解清楚对称的意思写错了，然后不知道怎么写递归函数，后来看了一下题解才意识到是需要传两个参数，然后左左=右右，左右=右左，想通了就好写了

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if(root == null) return true;
        
        return isSymmetric(root.left, root.right);
        
    }

    public boolean isSymmetric(TreeNode left, TreeNode right){
        if(left == null && right == null) return true;
        if(left == null || right == null) return false;
        if(left.val == right.val){
            return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
        }
        return false; // 注意不要漏掉val不相等的情况
    }
}
```

时间复杂度O(N) 空间复杂度O(N)



#### 29 顺时针打印矩阵

做过好几次的题目了，还是有点不熟练

```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        int[][] directions = {{0,1}, {1,0}, {0,-1}, {-1,0}};
        int x = 0, y = 0;
        if(matrix.length == 0 || matrix[0].length == 0) return new int[0];
        int height = matrix.length;
        int width = matrix[0].length;
        int[] res = new int[height * width];
        boolean[][] visited = new boolean[height][width];
        int direction = 0;
        int nextRow, nextCol;
        for(int i = 0; i < res.length; i++){
            visited[x][y] = true;
            res[i] = matrix[x][y];
            // 写到这里忘记怎么做了，去看了眼题解
            nextRow = x + directions[direction][0];
            nextCol = y + directions[direction][1];
            if(nextRow < 0 || nextRow >= height || nextCol < 0 || nextCol >= width || visited[nextRow][nextCol]){
                direction = (direction + 1) % 4;
            }
            x = x + directions[direction][0];
            y = y + directions[direction][1];
        }
        return res;
    }
}
```

设置了方向数组，比较重要的是每走到一个点时标记为走过并且加进结果数组（如果不满足要求就不会走到这个点），然后计算接下来要走的点是否符合要求，如果不符合就改变方向，得出下一个有效的点。



#### 30 包含min函数的栈（单调栈）

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

思路有点难想，**要维护两个栈**，其中一个用来保存每个时刻的最小值，例如 7 8 6 9 5 4对应的最小值栈就是7 6 5 4，当pop出去的值等于最小值栈的栈顶时，最小值栈也执行一次pop把这个最小值移出去，不在最小值栈中的数在任何时刻都不可能是最小值

一开始想到的只是维护一个最小值，当最小值pop出去之后就不知道该怎么办了，其实应该联想到应该维护所有可能成为最小值的数

```java
class MinStack {

    /** initialize your data structure here. */
    Stack<Integer> stack;
    Stack<Integer> minstack;
    public MinStack() {
        stack = new Stack<>();
        minstack = new Stack<>();
    }
    
    public void push(int x) {
        if(minstack.isEmpty() || x <= minstack.peek()){
            minstack.push(x);
        }
        stack.push(x);
    }
    
    public void pop() {
        if(stack.peek().equals(minstack.peek())) minstack.pop();
        stack.pop();
    }
    
    public int top() {
        return stack.peek();
    }
    
    public int min() {
        return minstack.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```

**注意：**Integer类型比较要用equals() !!! 一开始用==所以结果错了



#### 31 栈的压入弹出序列（模拟）

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

一开始觉得很难，其实还好，不要被题目吓到

true的情况就是最后栈为空

false的情况就是该弹出某个值的时候栈顶不是这个值

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int pos = 0;
        for(int i = 0; i < pushed.length; i++){
            while(!stack.isEmpty() && stack.peek() == popped[pos]){
                stack.pop();
                pos++;
            }
            stack.push(pushed[i]);
        }
        while(!stack.isEmpty()){
            if(stack.peek() == popped[pos]){
                stack.pop();
                pos++;
            }
            else{
                return false;
            }
        }
        
        return true;
    }
}
```



#### 32-1 从上到下打印二叉树（层次遍历）

**注意** 第一次提交的时候没有考虑到root为null的情况

`toArray()`函数只能把`ArrayList<Integer>`转为`Integer[]`

注释掉的部分可以很方便地做到前面的循环做的事情，但是慢了挺多

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if(root == null) return new int[0];
        ArrayList<Integer> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode curr = queue.poll();
            res.add(curr.val);
            if(curr.left != null) queue.add(curr.left);
            if(curr.right != null) queue.add(curr.right);
        }
        int[] arr = new int[res.size()];
        for(int i = 0; i < arr.length; i++){
            arr[i] = res.get(i);
        }
        //int[] arr = res.stream().mapToInt(i -> i).toArray();
        return arr;
    }
}
```



#### 32-2 从上到下分层打印二叉树

给定二叉树: [3,9,20,null,null,15,7],

```
    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：
[
  [3],
  [9,20],
  [15,7]
]
```

使用`queue.size()`来分层

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null) return res; 
        queue.add(root);
        while(!queue.isEmpty()){
            int cnt = queue.size();
            List<Integer> list = new ArrayList<>();
            while(cnt > 0){
                TreeNode curr = queue.poll();
                list.add(curr.val);
                if(curr.left != null) queue.add(curr.left);  
                if(curr.right != null) queue.add(curr.right);    
                cnt--;
            }
            res.add(list);
        }
        return res;
    }
}
```



#### 32-3 从上到下打印二叉树(分层+之字形)

其实很简单，但是找了半天才看出来粗心地把direction的赋值写在了循环内部，这样每次都会给direction赋值为true

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        if(root == null) return res; 
        queue.add(root);
        boolean direction = true;//要在这里赋值！！！！
        while(!queue.isEmpty()){
            int cnt = queue.size();
            List<Integer> list = new ArrayList<>();
            while(cnt > 0){
                TreeNode curr = queue.poll();
                list.add(curr.val);
                if(curr.left != null) queue.add(curr.left);  
                if(curr.right != null) queue.add(curr.right);    
                cnt--;
            }
            if(!direction){
                Collections.reverse(list);
            } 
            res.add(list);
            direction = !direction;
        }
        return res;
    }
}
```



#### 33 二叉搜索树的后序遍历序列

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。参考以下这颗二叉搜索树：

```
     5
    / \
   2   6
  / \
 1   3
 
输入: [1,6,3,2,5]
输出: false

输入: [1,3,2,6,5]
输出: true
```

一开始很懵，不明白要怎么判断，后来看了一眼题解的图

其实和之前的做法差不多，后序遍历是左右中，根节点一定是最后一个节点，左边都比根节点小，右边都比根节点大，先根据这三个规则找到左右的分界点，再递归判断

注意不要漏掉没有左子树或者右子树的情况

中间浪费了很多时间在编译出错上，都是非常粗心的错。

1. 递归调用函数时第二个函数忘记写函数名
2. 新写的函数里忘记给参数设置为int类型，只写了参数名字

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        int len = postorder.length;
        if(len == 0 || len == 1) return true;
        return verifyPostorder(postorder, 0, len - 1);
    }
    public boolean verifyPostorder(int[] postorder, int left, int right){
        if(left >= right) return true;
        int partition = left;
        int root = postorder[right];
        while(partition < right){
            if(postorder[partition] > root){
                break;
            }
            partition++;
        }
        for(int i = partition; i < right; i++){
            if(postorder[i] < root) return false;
        }
        return verifyPostorder(postorder, left, partition - 1) && verifyPostorder(postorder, partition, right - 1);
    }
}
```



#### 34 二叉树中和为某一值的路径

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

自己写的递归如下：

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int target) {
        List<Integer> list = new ArrayList<>();
        pathSum(root, target, list);
        return res;
    }
    public void pathSum(TreeNode root, int target, List<Integer> list){
        if(root == null) return;
        if(root.left == null && root.right == null){
            if(target == root.val){
                list.add(root.val);
                res.add(new ArrayList<Integer>(list));
                //注意这里要new一个新的，否则res里的元素都是同一个list的引用
                list.remove(list.size() - 1);
                return;
            }
            else{
                return;
            }
        }
        list.add(root.val);
        pathSum(root.left, target - root.val, list);
        pathSum(root.right, target - root.val, list);
        list.remove(list.size() - 1);
    }
}
```

**注意**：java传list传的是引用！！！放结果的时候要new，add之后要remove，list在函数调用的时候是共用的

list.add的返回值是布尔型



#### 35 复杂链表的复制

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

一开始没有头绪，看了一下题解才发现其实很简单

就是用一个map存储原来的节点和复制出来的节点，用两轮循环，第一轮只建立结点和拷贝值，第二轮已经有全部结点了就可以拷贝next和random

时间和空间复杂度都是O(N)

```java
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Map<Node, Node> map = new HashMap<>();

        Node curr = head;
        while(curr != null){
            map.put(curr, new Node(curr.val));
            curr = curr.next;
        }
        curr = head;
        while(curr != null){
            map.get(curr).next = map.get(curr.next);
            map.get(curr).random = map.get(curr.random);
            curr = curr.next;
        }
        return map.get(head);
    }
}
```



#### 36 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

刚看到可能觉得有点难，其实并不难，只需要理清楚递归函数的返回值，我没看题解自己写的如下：

```java
class Solution {
    public Node treeToDoublyList(Node root) {
        if(root == null) return null;//记得判空
        treeToList(root);
        Node first = getFirst(root);
        Node last = getLast(root);
        first.left = last;
        last.right = first;
        return first;
    }
    public Node getFirst(Node root){
        if(root == null) return null;
        if(root.left == null) return root;
        else{
            return getFirst(root.left);
        }
    }
    public Node getLast(Node root){
        if(root == null) return null;
        if(root.right == null) return root;
        else{
            return getLast(root.right);
        }
    }
    public void treeToList(Node root){
        if(root.left != null){
            Node left = getLast(root.left);//注意这里要先把left求出来存好
            treeToList(root.left);
            root.left = left;
            left.right = root;
        } 
        if(root.right != null){
            Node right = getFirst(root.right);
            treeToList(root.right);
            root.right = right;
            right.left = root;
        }         
    }
}
```

时间和空间击败100%和83%

看了题解，题解的做法更加简洁，可背

使用中序遍历访问结点，访问的时候构建cur和pre的指向，cur就是当前中序遍历到的点

```java
class Solution {
    Node pre, head;
    public Node treeToDoublyList(Node root) {
        if(root == null) return null;
        dfs(root);
        head.left = pre;
        pre.right = head;
        return head;
    }
    void dfs(Node cur) {
        if(cur == null) return;
        dfs(cur.left);
        if(pre != null) pre.right = cur;
        else head = cur;
        cur.left = pre;
        pre = cur;
        dfs(cur.right);
    }
}

作者：jyd
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/solution/mian-shi-ti-36-er-cha-sou-suo-shu-yu-shuang-xian-5/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



#### 37 序列化二叉树（层次遍历正+反，hard题）

需要写两个函数，一个是序列化一个是反序列化。题目对序列化的方式不做要求，只要两个函数可逆即可。我使用的是层次遍历的做法。序列化比较简单，但是反序列化（把string变成tree）思路很乱，看了题解发现很巧妙。其实写法十分类似层次遍历，也是用一个队列来存树的结点，不同的是这里用了一个i来指示目前遍历到字符串列表中的哪一个元素，i每次加一就可以了，非常巧妙，i总是指向当前结点的子结点

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        List<String> list = new ArrayList<>();
        while(!queue.isEmpty()){
            TreeNode curr = queue.poll();
            if(curr == null){
                list.add("null");
                continue;
            } 
            list.add(String.valueOf(curr.val));        
            queue.add(curr.left);
            queue.add(curr.right);
        }
        String[] strs = list.toArray(new String[list.size()]);
        String res = Arrays.toString(strs);
        return res;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] strs = data.substring(1, data.length() - 1).split(", ");
        if(strs[0].equals("null")) return null;
        TreeNode root = new TreeNode(Integer.parseInt(strs[0]));
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int i = 1;
        while(!queue.isEmpty()){
            TreeNode curr = queue.poll();
            if(curr == null){
                continue;
            }
            curr.left = (strs[i].equals("null")) ? null : new TreeNode(Integer.parseInt(strs[i]));
            queue.add(curr.left);
            i++;
            curr.right = (strs[i].equals("null")) ? null : new TreeNode(Integer.parseInt(strs[i]));
            queue.add(curr.right);
            i++;
        }
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```

有一个踩坑的点

**string比较是否相等要用equals不能用==**

还有一个踩坑的点

**java里面要把list转成的string转回list，split的参数是", "，是逗号加空格，split的参数必须是字符串不能是char**

```java
//语法
//int -> string
String.valueOf(curr.val)
//string -> int
Integer.parseInt(strs[0])
```



还可以使用前序遍历来序列化，利用递归函数，注意递归函数的参数和返回值

```java
public class Codec {
    public String serialize(TreeNode root) {
        return rserialize(root, "");
    }
  
    public TreeNode deserialize(String data) {
        String[] dataArray = data.split(",");
        List<String> dataList = new LinkedList<String>(Arrays.asList(dataArray));
        return rdeserialize(dataList);
    }

    public String rserialize(TreeNode root, String str) {
        if (root == null) {
            str += "None,";
        } else {
            str += str.valueOf(root.val) + ",";
            str = rserialize(root.left, str);//这里很重要
            str = rserialize(root.right, str);
        }
        return str;
    }
  
    public TreeNode rdeserialize(List<String> dataList) {
        if (dataList.get(0).equals("None")) {
            dataList.remove(0);//这里很重要
            return null;
        }
  
        TreeNode root = new TreeNode(Integer.valueOf(dataList.get(0)));
        dataList.remove(0);//这里很重要
        root.left = rdeserialize(dataList);
        root.right = rdeserialize(dataList);
    
        return root;
    }
}
```



#### 38 字符串的排列（经典+变形backtracking）

输入一个字符串，打印出该字符串中字符的所有排列。输入的字符串可含有重复字符，例如：aab

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

比起普通的全排列多了一个限制，我参考了”搜索.md”写的如下

思路是，先给字符排序，若当前遍历的字符和前面的相同而且前面的字符hasVisited为false，则跳过这个字符，例如a a a，用这种思路就可以保证只有一种排列（a1, a2, a3）

```java
class Solution {
    public String[] permutation(String s) {        
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        boolean[] hasVisited = new boolean[chars.length];
        StringBuilder strb = new StringBuilder();
        List<String> list = new ArrayList<>();
        permutation(chars, hasVisited, strb, list);
        String[] res = list.toArray(new String[list.size()]);
        return res;
    }
    public void permutation(final char[] chars, boolean[] hasVisited, StringBuilder strb, List<String> permuteList){
        if(strb.length() == chars.length){
            permuteList.add(strb.toString());
            return;
        }
        for(int i = 0; i < chars.length; i++){
            if(hasVisited[i] || (i > 0 && !hasVisited[i - 1] && chars[i] == chars[i - 1])){
                continue;
            }
            strb.append(chars[i]);
            hasVisited[i] = true;
            permutation(chars, hasVisited, strb, permuteList);
            hasVisited[i] = false;
            strb.deleteCharAt(strb.length() - 1);
        }
    }
}
```

别人写的题解如下：

```java
class Solution {
    List<String> res = new LinkedList<>();
    char[] c;
    public String[] permutation(String s) {
        c = s.toCharArray();
        dfs(0);
        return res.toArray(new String[res.size()]);
    }
    void dfs(int x) {
        if(x == c.length - 1) {
            res.add(String.valueOf(c));      // 添加排列方案
            return;
        }
        HashSet<Character> set = new HashSet<>();
        for(int i = x; i < c.length; i++) {
            if(set.contains(c[i])) continue; // 重复，因此剪枝
            set.add(c[i]);
            swap(i, x);                      // 交换，将 c[i] 固定在第 x 位
            dfs(x + 1);                      // 开启固定第 x + 1 位字符
            swap(i, x);                      // 恢复交换
        }
    }
    void swap(int a, int b) {
        char tmp = c[a];
        c[a] = c[b];
        c[b] = tmp;
    }
}
```

两种做法的思路还是有些差别的，法二使用了一个集合（里面的元素不重合）来帮助剪枝，若 `c[i]` 在 Set 中，代表其是重复字符，因此 “剪枝“，还使用了一个x用来表示固定的位数

我更建议第一种，更规整不容易出错



#### 39 数组中出现次数超过一半的数字（摩尔投票法）

一开始用了最容易想到的map。时间和空间表现都不太好。后来看了题解，感觉第二种很妙，只要两行

> 题中说了给定的数组总是存在多数元素。，也就是说肯定有一个元素的个数大于数组长度的一半。我们只需要把这个数组排序，那么数组中间的值肯定是存在最多的元素。
>
> 其实很容易证明，假设排序之后数组的中间值不是最多的元素，那么这个最多的元素要么是在数组前半部分，要么是在数组的后半部分，无论在哪，他的长度都不可能超过数组长度的一半。
>

```java
class Solution {
    public int majorityElement(int[] nums) {
        // 第一种
        // int len = nums.length;
        // Map<Integer, Integer> map = new HashMap<>();
        // for(int i = 0; i < len; i++){ 
        //     map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        //     if(map.get(nums[i]) > len / 2) return nums[i];            
        // }
        // return 0;
		
        // 第二种
        // Arrays.sort(nums);
        // return nums[nums.length / 2];
        
        // 第三种
        int res = 0, sum = 0;
        for(int i = 0; i < nums.length; i++){
            if(sum == 0) res = nums[i];
            if(nums[i] == res) sum++;
            else sum--;
        }
        return res;
    }
}
```

比较难懂的是第三种，核心理念是票数正负抵消，假定第一个是众数，在票数和为0时更新众数猜测值。



#### 40 最小的k个数

输入整数数组 `arr` ，找出其中最小的 `k` 个数。

**限制：**

- `0 <= k <= arr.length <= 10000`
- `0 <= arr[i] <= 10000`

因为数据范围有限，我用了桶排序的思想做的，时间复杂度O(N)，打败99%

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0 || arr.length == 0) {
            return new int[0];
        }
        int[] counter = new int[10001];
        for (int num: arr) {
            counter[num]++;
        }
        int[] res = new int[k];
        int j = 0;
        for(int i = 0; i < counter.length; i++){
            while(counter[i] > 0){
                res[j] = i;
                counter[i]--;
                j++;
                if(j == k) return res;
            }
        }
        return res;
    }
}
```

更万能的做法是利用快排，但是对于这道题还是用桶更好

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if(k == 0 || arr.length == 0) return new int[0];
        int low = 0, high = arr.length - 1;
        int pivot = partition(arr, low, high);
        while(pivot != k - 1){
            if(pivot > k - 1){
                high = pivot - 1; //必须是-1否则会死循环，参考arr=[3,2,1]              
            }
            else{
                low = pivot + 1;//同理
            }
            pivot = partition(arr, low, high);
        }
        return Arrays.copyOf(arr, k);
    }
    public int partition(int[] nums, int low, int high){
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
}
```



#### 41 数据流中的中位数（大顶堆+小顶堆）

> 设计一个支持以下两种操作的数据结构：
>
> void addNum(int num) - 从数据流中添加一个整数到数据结构中。
> double findMedian() - 返回目前所有元素的中位数。

不难，就是一开始少考虑了一种情况，即元素不能随便地选一个堆插入，要看它的值，正确插入到对的那一半之后再去让两边的个数保存平衡

还有返回结果时也要相应地修改，这个好理解

```java
class MedianFinder {
    Queue<Integer> queue_min;
    Queue<Integer> queue_max;
    /** initialize your data structure here. */
    public MedianFinder() {
        queue_min = new PriorityQueue<Integer>();
        queue_max = new PriorityQueue<Integer>(new Comparator<Integer>(){
            public int compare(Integer o1, Integer o2){
                return o2 - o1;
            }
        });
    }
    
    public void addNum(int num) {
        if(!queue_max.isEmpty() && queue_max.peek() > num){
            queue_max.add(num);
            if(queue_max.size() - queue_min.size() == 2){
                queue_min.add(queue_max.poll());
            }
        }
        else{
            queue_min.add(num);
            if(queue_min.size() - queue_max.size() == 2){
                queue_max.add(queue_min.poll());
            }
        }
        
    }
    
    public double findMedian() {
        if(queue_min.size() == queue_max.size()) 
            return (queue_min.peek() + queue_max.peek()) / 2.0;
        else if(queue_min.size() > queue_max.size())
            return queue_min.peek();
        else return queue_max.peek();
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```



#### 42 连续子数组的最大和（很有趣的DP）

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

一开始完全想不到要怎么dp，后来看了题解发现这题以前做过，可恶居然忘了！

> dp[i] 代表以元素 nums[i] 为结尾的连续子数组最大和，对dp[i-1]是否大于0进行分类讨论

```java
class Solution {
    public int maxSubArray(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        dp[0] = nums[0];
        int max = dp[0];
        for(int i = 1; i < len; i++){
            if(dp[i - 1] < 0) dp[i] = nums[i];
            else dp[i] = dp[i - 1] + nums[i]; 
            max = Math.max(max, dp[i]);
        }
        return max;
    }
}
```



#### 43 1-n整数中1出现的次数（hard，很难，头很晕）

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

看了题解叹为观止

最后自己推了一遍

curr表示当前位，digit表示当前位是个十百千等，digit=1表示curr是个位，digit=10表示curr是十位

high是curr前面的部分，low是curr后面的部分

一个数字就表示为[high]\[curr]\[low]=(high\*10+curr)*digit+low

```java
class Solution {
    public int countDigitOne(int n) {
        int digit = 1, low = 0, high = n / 10, curr = n % 10;
        int res = 0;
        //注意循环终止条件
        while(high != 0 || curr != 0){
            //下面三行是找出的规律
            //分为当前位比1小，等于1，大于1
            //如果是求其他数字出现的次数就按其他数字来分割
            if(curr == 0){
                res += high * digit;
            }
            else if(curr == 1){
                res += high * digit + low + 1;
            }
            else{
                res += (high + 1) * digit;
            }
            //注意变量变化规律
            low += curr * digit;
            digit *= 10;
            curr = high % 10;
            high /= 10;  
        }
        return res;
    }
}
```

每次只需要对n的某一位进行计算，非常厉害，时间复杂度为O(log n)，就是n的位数。空间复杂度为O(1)

**至此，我终于把leetcode上的剑指offer题做完了一遍，接下来打算转战leetcode前一百题**



#### 44 数字序列中某一位的数字（因为long写了半天的题

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

请写一个函数，求任意第n位对应的数字。

`0 <= n < 2^31`

这道题算是找规律，首先第0个数单独考虑，接下来有9个1位数，90个两位数，900个三位数···而且1位数从1开始，两位数从10开始，三位数从100开始，每个区间的长度就可以写成9 x 开始的数 x 每个数有几位。然后看第n位数在哪一个区间，通过一个while循环找出来，还可以知道它在这个区间的第几位

接下来要根据它在区间的第几位和这个区间的数字都有几位找出它所在的数字，我写得有点冗长，但这就是我的思路，题解里写得更简洁，但是我想不到这种写法，也不太明白原理。题解里`num = start + (n - 1) / digit` ,   `s.charAt((n - 1) % digit) - '0'`

```java
class Solution {
    public int findNthDigit(int n) {
        if(n == 0) return 0;
        int digit = 1;
        long start = 1, count = 9 * start * digit;
        while(n > count){           
            n -= count;
            start *= 10;
            digit++;
            count = 9 * start * digit;              
        }
        long num = start + n / digit - 1;
        if(n % digit != 0) num++;
        String s = String.valueOf(num);
        return s.charAt((n % digit + digit - 1) % digit) - '0';
    }
}
```

这道题不仅后面的找规律，加一减一取整花了很多时间，long和int也浪费了很多时间，一开始根本没想通错在哪，后来发现题解里的digit必须为int，start为long，才可以没有写强制类型转换。然而题解里只字未提这个，所以浪费了很多时间

后来看了题解下面的评论

> 对于第二步：这里取n-1的原因是:当n对应num中的最后一位时，不会由于进位，错误的寻找到下一个数字。
>
> 对于第三步：这里取n-1使得num中各位的位置坐标从左到右从0开始增加，即0,1,....,(digit-1)，更容易理解。
>
> 另外，这里的java代码把n=0单独提出来可能更清晰一点，而且start的值应该一直小于n的值，n为int，start应该用int的范围就够了，只是count的值在第一步最后一次循环时可能会超限，应该用long。最后确定结果的时候也可以不用转为字符数组，下面是相关代码和注释：

```java
 public int findNthDigit(int n) {
        if(n==0) return 0;
        //由于是n=0时对应开始的0，这里不需要进行减操作n--;，但是如果n=1对应开始的0则需要减操作
        //排除n=0后，后面n从1开始。
        int digit = 1;
        int start = 1;
        long count = 9; //count的值有可能会超出int的范围，所以变量类型取为long
        while(n>count){//不能带=号，此时n从1开始，n=count时，属于上一个循环的最后一个数字
            n=(int)(n-count);//这里(int)不能省略
            digit++;
            start = start*10;
            count = (long)start*9*digit;
            //这里的long不能省略，否则，会先按照int类型进行计算，然后赋值给long型的count，超过int大小限制时，会出现负数
        }

        int num = start + (n-1)/digit;
        int index = (n-1)%digit;//index最大取digit-1,即此时num坐标从左到右为0,1,...,digit-1,共digit位
        while(index<(digit-1)){
        //最后的结果是num中的第index个数字，index从左到右从0开始递增，考虑到踢出右侧末尾的数字比较简单，我们从右侧开始依次踢出数字
            num = num/10;
            digit--;
        }
        return num%10;//此时num的右侧末尾数字即为结果
    }
```



#### 45 把数组排成最小的数

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

```
输入: [10,2]
输出: "102"
输入: [3,30,34,5,9]
输出: "3033459"
```

比想象的要简单，**最重要的是**重新定义排序/比较的规则

如果AB>BA，则A大于B，所以要把B排在A前面

还注意相关语法

```java
class Solution {
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i = 0; i < nums.length; i++){
            strs[i] = Integer.toString(nums[i]);
        }
        Arrays.sort(strs, new Comparator<String>(){
            public int compare(String o1, String o2){
                return (o1 + o2).compareTo(o2 + o1);
            }
        });
        StringBuilder res = new StringBuilder();
        for(int i = 0; i < nums.length; i++){
            res.append(strs[i]);
        }
        return res.toString();
    }
}
```



#### 46 把数字翻译成字符串（被气到捶桌子的动归）

气死我了气死我了好气好气好气好气

> 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
>

+ int可以直接用函数转成String
+ String可以用compareTo和 “26” 之类的数字字符串进行比较
+ f(i)表示的是以i结尾时有几种，一开始只想到递归的写法，死活想不到这个思路
+ 注意0是可以翻译的，一开始一直以为不行
+ 09这种是不行的
+ f(i) = f(i-1)+f(i-2)(i-1到 i 的子串>=10且<26)
+ pre2的初始值为1不是0，以25为例，这个有点绕

```java
class Solution {
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int pre2 = 1, pre1 = 1, curr = 1;
        for(int i = 1; i < s.length(); i++){
            curr = pre1;
            if(s.substring(i - 1, i + 1).compareTo("26") < 0 && s.substring(i - 1, i + 1).compareTo("10") >= 0){
                curr += pre2;
            }
            pre2 = pre1;
            pre1 = curr;
        }
        return curr;
    }
}
```



#### 47 礼物的最大价值（简单动归）

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

第一次运行出错是因为在初始化两条边界的时候直接赋了原来的值，应该要加上它的前一个值的，改完就好了

```java
class Solution {
    public int maxValue(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;
        int[][] res = new int[row][col];
        res[0][0] = grid[0][0];
        for(int i = 1; i < row; i++){
            res[i][0] = grid[i][0] + res[i - 1][0];//这里一开始忘记加前面算出来的值了
        }
        for(int i = 1; i < col; i++){
            res[0][i] = grid[0][i] + res[0][i - 1];//这里一开始忘记加前面算出来的值了
        }
        for(int i = 1; i < row; i++){
            for(int j = 1; j < col; j++){
                res[i][j] = grid[i][j] + Math.max(res[i - 1][j], res[i][j - 1]);
            }
        }
        return res[row - 1][col - 1];
    }
}
```

所以其实这种题额外加两条赋值为0的边界可能会更不容易出错吧，但我不太喜欢下标转换



#### 48 最长不含重复字符的子字符串

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

```
输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
```

我用了hash set加滑动窗口，不难，但是第一次提交出错，因为没有考虑到输入的可以为空串，把答案的初始值改为0就好了

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int res = 0;//这里不能写1，要考虑输入为空串
        int start = 0;
        Set<Character> set = new HashSet<Character>();
        for(int i = 0; i < s.length(); i++){
            while(set.contains(s.charAt(i))){
                set.remove(s.charAt(start));
                start++;
            }
            set.add(s.charAt(i));
            res = Math.max(set.size(), res);
        }
        return res;
    }
}
```



#### 49 丑数（经典题，做过又忘，多路归并）

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```

一开始我的做法是从6开始判断每个数除以2，3，5之后的结果是不是丑数，结果超时了，想想确实，越到后面间隔越大，这么做很浪费时间，代码如下：

```java
class Solution {
    public int nthUglyNumber(int n) {
        Set<Integer> set = new HashSet<Integer>();
        set.add(1);
        set.add(2);
        set.add(3);
        set.add(4);
        set.add(5);
        if(n <= 5) return n;
        int count = 5;
        int i = 6;
        while(true){
            if((set.contains(i / 2) && i % 2 == 0) || (i % 3 == 0 && set.contains(i / 3)) || (i % 5 == 0 && set.contains(i / 5))){
                count++;
                if(count == n) return i;
                set.add(i);
            }
            i++;
        }
    }
}
```

后来还是看了题解，才想起来做法。而且还看了好几遍题解。。。

用了三个指针，初始都指向第一个丑数，计算出它的2，3，5倍数，选出最小的那个，相应的指针就可以**指向下一个丑数**。如果计算结果有相同的，就把多个指针都指向下一个丑数

**最佳做法**

```java
class Solution {
    public int nthUglyNumber(int n) {
        int p2 = 0, p3 = 0, p5 = 0;
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i = 1; i < n; i++){
            int num2 = 2 * dp[p2];
            int num3 = 3 * dp[p3];
            int num5 = 5 * dp[p5];
            dp[i] = Math.min(num2, Math.min(num3, num5));
            if(dp[i] == num2) p2++;
            if(dp[i] == num3) p3++;//这里要注意，不要用else if，因为有时候是num2==num3
            if(dp[i] == num5) p5++;//同上，通过这种方式处理重复
        }
        return dp[n - 1];
    }
}
```

还有一种做法是用最小堆（选最小的数）加集合（去重），这样做的时间复杂度是O(NlogN)，前一种是O(N)

```java
class Solution {
    public int nthUglyNumber(int n) {
        if(n == 1) return 1;
        Set<Long> set = new HashSet<Long>();//必须是long，因为后面add会把计算到的即使不是最小的也放进去，如果超过int范围就会变成负数，然后被最小堆选出来
        Queue<Long> queue = new PriorityQueue<Long>();
        set.add(1L);
        queue.add(1L);
        int[] nums = {2, 3, 5};
        for(int i = 1; i < n; i++){
            long curr = queue.poll();
            for(int num : nums){
                long next = curr * num;
                if(set.add(next)){//如果set里面已经有了，就不会加到queue里了
                    queue.add(next);
                }
            }
        }
        return (int)queue.poll().longValue();//必须这么转才行
    }
}
```



#### 50 第一个只出现一次的字符

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母

我的思路：因为只有小写字母，所以只需要26长度的int数组，数组里存遍历到字符时的位置，初始都是0，如果第1次遇到，就存位置，超过第一次就置为-1

最后再扫描一遍找出大于0的最小的那个。

**注意：** char->int可以直接转，int->char需要(char)('a'+1)

```java
class Solution {
    public char firstUniqChar(String s) {
        int[] positions = new int[26];
        for(int i = 0; i < s.length(); i++){
            int tmp = s.charAt(i)-'a';
            if(positions[tmp] == 0){
                positions[tmp] = i + 1;
            }
            else if(positions[tmp] > 0){
                positions[tmp] = -1;
            }
        }
        int min = s.length() + 1;
        char res = ' ';
        for(int i = 0; i < 26; i++){
            if(positions[i] > 0 && positions[i] < min){
                min = positions[i];
                res = (char)('a' + i);
            } 
        }
        return res;
        
    }
}
```



#### 51 数组中的逆序对（经典常考题）

这道题做过很多次，但是之前都是用的暴力解法

看了题解，用的是归并排序的思想，简单地修改了归并排序的函数就可以了

> 把 `lPtr` 对应的数加入答案，并考虑它对逆序对总数的贡献为 `rPtr` 相对 R*R* 首位置的偏移
>
> 用这种「算贡献」的思想在合并的过程中计算逆序对的数量的时候，只在 lPtr 右移的时候计算，是基于这样的事实：当前 lPtr 指向的数字比 rPtr 小，但是比 RR 中 [0 ... rPtr - 1] 的其他数字大，[0 ... rPtr - 1] 的其他数字本应当排在 lPtr 对应数字的左边，但是它排在了右边，所以这里就贡献了 rPtr 个逆序对。
>

做了才发现**归并排序还是不够熟练**，重点是merge的写法

```java
class Solution {
    public int reversePairs(int[] nums) {
        return reversePairs(nums, 0, nums.length - 1);
    }
    public int reversePairs(int[] nums, int left, int right){
        if(left >= right) return 0;
        int mid = left + (right - left) / 2;
        int countLeft = reversePairs(nums, left, mid);
        int countRight = reversePairs(nums, mid + 1, right);       
        if(nums[mid] <= nums[mid + 1]) return countLeft + countRight;
        int countBoth = merge(nums, left, mid, right);
        return countLeft + countRight + countBoth;
    }
    public int merge(int[] nums, int left, int mid, int right){
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        int count = 0;
        while(i <= mid && j <= right){
            if(nums[i] <= nums[j]){
                temp[k] = nums[i];
                i++;
                count += j - mid - 1;//这里是修改归并排序的部分
            }
            else{
                temp[k] = nums[j];
                j++;
            }
            k++;
        }
        while(i <= mid){
            temp[k] = nums[i];
            i++;
            count += j - mid - 1;
            k++;//这里漏写了，找了一上午Bug
        }
        while(j <= right){
            temp[k] = nums[j];
            j++;
            k++;//这里漏写了，找了一上午Bug
        }
        for(i = left; i <= right; i++){
            nums[i] = temp[i - left];
        }//这步很重要，要修改nums的值
        return count;
    }
}
```

另一种思路是https://mp.weixin.qq.com/s/3mksg14RLc15BhKuAR5oHA这个题解里看的，我觉得这个会比官方题解的思路更好懂。

做法是在merge的时候，如果当前的left大于当前的right，意味着 「左子数组当前元素 至 末尾元素」 与 「右子数组当前元素」 构成了若干 「逆序对」。

代码如下

```java
class Solution {
    public int reversePairs(int[] nums) {
        return reversePairs(nums, 0, nums.length - 1);
    }
    public int reversePairs(int[] nums, int left, int right){
        if(left >= right) return 0;
        int mid = left + (right - left) / 2;
        int countLeft = reversePairs(nums, left, mid);
        int countRight = reversePairs(nums, mid + 1, right);       
        if(nums[mid] <= nums[mid + 1]) return countLeft + countRight;
        int countBoth = merge(nums, left, mid, right);
        return countLeft + countRight + countBoth;
    }
    public int merge(int[] nums, int left, int mid, int right){
        int[] temp = new int[right - left + 1];
        int i = left, j = mid + 1, k = 0;
        int count = 0;
        while(i <= mid && j <= right){
            if(nums[i] <= nums[j]){
                temp[k] = nums[i];
                i++;
            }
            else{
                temp[k] = nums[j];
                j++;
                // 和普通merge比只要改一行
                count += mid - i + 1;
            }
            k++;
        }
        while(i <= mid){
            temp[k] = nums[i];
            i++;
            k++;
        }
        while(j <= right){
            temp[k] = nums[j];
            j++;
            k++;
        }
        for(i = left; i <= right; i++){
            nums[i] = temp[i - left];
        }
        return count;
    }
}
```





#### 52 两个链表的第一个公共节点（经典链表题）

和leetcode第160题相同，这次做的时候记得思路但是忘记写法了

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode iterA = headA;
        ListNode iterB = headB;
        while(iterA != iterB){
            if(iterA == null) iterA = headB;
            else iterA = iterA.next;
            if(iterB == null) iterB = headA;
            else iterB = iterB.next;
        }
        return iterA;
    }
}
```



#### 53-1 在排序数组中查找数字（二分查找，有重复元素最左）

统计一个数字在排序数组中出现的次数。

```
示例 1:
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2

示例 2:
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

主要是画图考虑清楚每种情况，还有避免死循环，弄清楚循环结束的时候是什么，可以先把基础的二分查找写出来，然后进行修改。这里的left<right作为循环条件是为了避免死循环，循环结束后需要判断left/right是不是想要的值。

```java
class Solution {
    public int search(int[] nums, int target) {
        if(nums.length == 0) return 0;
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while(left < right){
            mid = left + (right - left) / 2;
            if(nums[mid] < target){
                left = mid + 1;
            }
            else if(nums[mid] > target){
                right = mid - 1;
            }
            else{
                right = mid;
            }
        }
        if(nums[left] != target) return 0;
        else{
            int res = 0;
            while(left < nums.length && nums[left] == target){
                res++;
                left++;
            }
            return res;
        }
    }
}
```

**边界情况**：传入的数组为[]或[1]，需要注意防止数组越界



#### 53-2 0~n-1中缺失的数字（二分查找经典题）

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

```
输入: [0,1,3]
输出: 2
```

```
输入: [0,1,2,3,4,5,6,7,9]
输出: 8
```

一开始不知道怎么二分，看了题解才知道怎么做，但和题解的做法有细微的不同。大体都是，把数组分成两部分，左侧是nums[m]==m，右侧是nums[m]!=m。要找到的是**右侧部分第一个数的索引**

还需要注意边界情况：第一种是数组只有一个元素，**第二种是要找的数字在边界**，例如[0,1]这种

```java
class Solution {
    public int missingNumber(int[] nums) {
        if(nums.length == 1){
            if(nums[0] == 0) return 1;
            else return 0;
        }
        int left = 0, right = nums.length - 1;
        int mid = 0;
        while(left < right){
            mid = left + (right - left) / 2;
            if(nums[mid] == mid) left = mid + 1;
            else right = mid;
        }
        if(nums[left] == left) return left + 1;
        return left;
    }
}
```

题解的做法会更简洁一点，两种思路的不同在于right=mid还是mid-1，mid-1的话表示right结束时指向左子数组的最后一个数字

```java
class Solution {
    public int missingNumber(int[] nums) {
        int i = 0, j = nums.length - 1;
        while(i <= j) {
            int m = (i + j) / 2;
            if(nums[m] == m) i = m + 1;
            else j = m - 1;
        }
        return i;
    }
}
```



#### 54 二叉搜索树的第k大结点（后序遍历+提前返回和终止）

这题思路不难，难的是不知道后序遍历的函数返回值怎么设置，到最后不得不看了一眼题解。才知道是需要两个全局变量，一个记录结果，一个记录现在遍历了多少个，返回值设为void，找到了第k个就设置res并且直接return。

```java
class Solution {
    int res = 0, current = 0;//!!!最重要的在这!!!
    public int kthLargest(TreeNode root, int k) {
        postOrder(root, k);
        return res;
    }
    public void postOrder(TreeNode root, int k){
        if(root.right != null) postOrder(root.right, k);
        current++;
        if(k == current){
            res = root.val;
            return;
        }
        if(root.left != null) postOrder(root.left, k);
    }  
}
```



#### 55-1 二叉树的深度

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if(root != null){
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
        }
        else return 0;
    }
}
```



#### 55-2 判断平衡二叉树（面试考过，有点绕）

最直观的思路是写两个函数，一个计算树的深度，一个利用左右子树的深度判断是不是平衡二叉树，代码如下：

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        if(root == null) return true;
        else{
            if(Math.abs(depth(root.left) - depth(root.right)) < 2 && isBalanced(root.left) && isBalanced(root.right)){
                return true;
            }
            return false;
        }
    }
    public int depth(TreeNode root) {
        if(root != null) {
            return Math.max(depth(root.left), depth(root.right)) + 1;
        }
        else return 0;
    }
}
```

但是这种做法没有利用到之前计算的结果，每个结点计算深度不止一次

看了题解，重点在于要用一个返回值表达两种意思（深度和是否平衡），这样可以避免重复计算

```java
class Solution {
    public boolean isBalanced(TreeNode root) {
        if(depth(root) == -1) return false;
        return true;
    }
    public int depth(TreeNode root) {
        if(root != null) {
            int left = depth(root.left);
            int right = depth(root.right);
            if(left != -1 && right != -1 && Math.abs(left - right) < 2){
                return Math.max(left, right) + 1;
            }
            else return -1;//不平衡就返回-1
        }
        else return 0;
    }
}
```

其实在看到这道题的时候，第一想法就是，我**在利用子树深度判断是否平衡时，也就知道了树的深度**，其实可以用一个函数解决，不需要写两个函数，但是没有想到利用-1表示不平衡，非负数表示平衡和深度



#### 56-1 数组中数字出现的次数（分组异或，难想）

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

```
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
```

这道题完全没有思路，所以看了题解

首先要知道，如果是要**找一个**只出现一次的数字，只需要把所有的数字异或起来就可以了，因为相同的数字异或之后为0，最终结果就是只出现一次的数字。现在是要找两个只出现一次的数字，就需要把数组分成两半，每半包含一个出现一次的数字。所以需要对数字分组。

因为把数字全部异或之后可以得到a异或b的值，所以只要选择a和b值不同的某一位，也就是a异或b的值中取1的位，按照这一位是0还是1来把数组分成两部分就行。

位运算很妙，但是很难想。

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int ret = 0;
        for(int num : nums){
            ret ^= num;
        }
        int div = 1;// div其实是一个类似0001000这样的mask
        while((div & ret) == 0){
            div <<= 1;
        }
        int a = 0, b = 0;
        for(int num : nums){
            if((num & div) == 0){
                a ^= num;
            }
            else{
                b ^= num;
            }
        }
        return new int[]{a, b};
    }
}
```



#### 56-2 数组中数字出现的次数（位运算，难想）

在一个数组 `nums` 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

思路是把每个数字都看成二进制表示，然后将它们二进制位的1分别相加，再对3取余，如果数字都是出现3次的话结果为0，所以取余结果为只出现一次的数字

具体是做法是用一个长度为32的数组来存放各个位上的1的个数之和。因此需要对每个数字取出它的每个位。将数字不断地进行无符号右移然后和1与运算即可得到该位为1还是0

> 利用 **左移操作** 和 **或运算** ，可将 counts数组中各二进位的值恢复到数字res上



```java
class Solution {
    public int singleNumber(int[] nums) {
        int[] counts = new int[32];
        for(int i = 0; i < nums.length; i++){
            for(int j = 0; j < 32; j++){
                counts[j] += nums[i] & 1;
                nums[i] >>>= 1;
            }
        }
        // 这个循环可以合并到下面的循环里
        // for(int j = 0; j < 32; j++){
        //     counts[j] %= 3;
        // }
        int res = 0;
        for(int j = 1; j <= 32; j++){
            // 必须先左移再或，否则最后一位或完再左移会超出去
            res <<= 1;
            res |= counts[32 - j] % 3;           
        }
        return res;
    }
}
```



#### 57 和为s的两个数字（超级经典）

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

这题最容易想到的就是用哈希表之类的存一下，扫描一遍找有没有和前面的数字和为target的。这样的时间和空间都是O(N)。

但是这题多了一个条件：数组有序，所以用双指针更快。思路是两数和大于target则右指针左移，反之左指针右移。

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left < right){
            int sum = nums[left] + nums[right];
            if(sum == target) break;
            else if(sum < target) left++;
            else right--;
        }
        return new int[]{nums[left], nums[right]};
    }
}
```



#### 57-2 和为s的连续正整数序列（滑动窗口）

这道题的要求是连续，所以可以用滑动窗口，求和还可以用等差数列求和公式，但我用前序和的思想速度更快，不需要乘除法

```java
class Solution {
    public int[][] findContinuousSequence(int target) {   
        int left = 1, right = 2;
        int sum = left + right;
        List<int[]> res = new ArrayList<int[]>();
        while(left < right){
            if(sum < target){
                right++;
                sum += right;
            }
            else if(sum > target){
                sum -= left;
                left++;//第一次这里写成了--
            }
            else{
                int[] curr = new int[right - left + 1];
                int start = left;
                for(int i = 0; i < curr.length; i++){
                    curr[i] = start;
                    start++;
                }
                res.add(curr);
                sum -= left;//第一次忘记写这两行了
                left++;//第一次忘记写这两行了
            }
        }
        return res.toArray(new int[res.size()][]);
    }
}
```

题解的代码更简练

```java
class Solution {
    public int[][] findContinuousSequence(int target) {
        List<int[]> vec = new ArrayList<int[]>();
        for (int l = 1, r = 2; l < r;) {
            int sum = (l + r) * (r - l + 1) / 2;
            if (sum == target) {
                int[] res = new int[r - l + 1];
                for (int i = l; i <= r; ++i) {
                    res[i - l] = i;
                }//这个for循环可以学习
                vec.add(res);
                l++;
            } else if (sum < target) {
                r++;
            } else {
                l++;
            }
        }
        return vec.toArray(new int[vec.size()][]);
    }
}
```



#### 58-1 翻转单词顺序（面试问过）

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

第一种做法就是用语言自带的函数，把分割出来的单词放进列表然后倒序取出来，但是如果不让用列表，就需要用题解看到的这个做法：

```java
class Solution {
    public String reverseWords(String s) {
        s = s.trim();
        int i = s.length() - 1, j = i;
        StringBuilder strb = new StringBuilder();
        while(i >= 0){
            while(i >= 0 && s.charAt(i) != ' ') i--;
            strb.append(s.substring(i + 1, j + 1) + " ");
            while(i >= 0 && s.charAt(i) == ' ') i--;
            j = i;
        }
        return strb.toString().trim();
    }
}
```

本质上是用两个指针i和j从后往前遍历，指向单词，append到StringBuilder上，这样就不需要另开一个列表了



#### 58-2 左旋转字符串（可以一行解决）

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

我写的是：

```java
class Solution {
    public String reverseLeftWords(String s, int n) {
        StringBuilder res = new StringBuilder();
        res.append(s.substring(n));
        res.append(s.substring(0, n));
        return res.toString();
    }
}
```

就挺简单的，其实还可以不用StringBuilder，直接`return s.substring(n) + s.substring(0, n)`就行了，两种做法速度都非常快，第二种更省内存。

如果面试的时候不让用substring函数，可以用charAt(i)一个一个的append到StringBuilder上

如果面试的时候不让用substring也不让用StringBuilder，就用`str1 += str0.charAt(i)`，但是这种做法效率最低，因为字符串是final类型



#### 59-1 滑动窗口的最大值（单调队列）

给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回滑动窗口中的最大值。

参考了[五分钟学算法](https://mp.weixin.qq.com/s?__biz=MzUyNjQxNjYyMg==&mid=2247498838&idx=2&sn=c10692783ae0bee1312ffe1aa7d10339&chksm=fa0d93d7cd7a1ac1ceff349bfe2a67770bdcebca3bd8a13c5148e6aacab087dbee8154cbeea8&scene=126&sessionid=1616233063&key=4764af25be59e44a2a6d0a22e84b56ef4f29cd5450586618381eda3dcc1a89ac4bc5c790ef342db8ed6bb61acafba342aa10f2ddac5900b381931c725a65f0db410c00813cad0ed142edc3df638b4c586d9a073685ba6c1f02a63c1f9e7c3e36cc53a2217df5d370a5a423bc947a7107213ecd9792c609729ee34dc8950465d8&ascene=1&uin=Mzg0Njg0NzU2&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A%2BQKkwMXJve5HGlpk1VN8u0%3D&pass_ticket=R%2F0%2Fb2w4jP8VfyFlSRsubdTmnhgWNUHpUDtMa%2FGe957YH6%2BYbTc3O%2FLJjZZMGFnF&wx_header=0)里的图解写出下面的代码

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length;
        if(len == 0) return new int[0];
        int[] res = new int[len - k + 1];
        LinkedList<Integer> queue = new LinkedList<>();
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

**注意**

输入的数组可能长度为0，需要增加一个判断，长度为零时返回[]



#### 59-2 队列的最大值（单调队列）

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

```java
class MaxQueue {
    LinkedList<Integer> queue;
    LinkedList<Integer> max_value;
    public MaxQueue() {
        queue = new LinkedList<>();
        max_value = new LinkedList<>();
    }
    
    public int max_value() {
        if(queue.isEmpty()) return -1;
        return max_value.peek();
    }
    
    public void push_back(int value) {
        queue.add(value);
        while(!max_value.isEmpty() && max_value.peekLast() < value){
            max_value.removeLast();
        }
        max_value.add(value);
    }
    
    public int pop_front() {
        if(queue.isEmpty()) return -1;
        if(queue.peek().equals(max_value.peek())){
            max_value.poll();
        }
        return queue.poll();
    }
}
```

**易错点**

+ 忘记处理队列为空的情况下pop和max
+ 忘记Integer的比较不能用==要用equals



#### 60 n个骰子的点数(有点难想的DP)

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。

```
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]

输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]
```

首先要能想到用DP做就好办很多了，可惜我没想到，是看了题解写的。dp\[n][x] = dp\[n-1][x-1]*1/6 + ... + dp\[n-1][x-6]\*1/6

DP的重点在于利用前一个概率分布求后一个概率分布。所以只需要用两个数组来存放数据即可。为了避免数组越界的麻烦，采用从前往后推的做法

```java
class Solution {
    public double[] dicesProbability(int n) {
        double[] dp = new double[6];//dp用来存前一个数组，tmp用来存后一个数组
        Arrays.fill(dp, 1.0 / 6.0);
        for(int i = 2; i <= n; i++){
            double[] tmp = new double[5 * i + 1]; //每个概率数组的长度为6n-n+1=5n+1
            for(int j = 0; j < dp.length; j++){ //对前一个数组的每个元素计算它对后一个数组的贡献
                for(int k = 1; k <= 6; k++){ //对多出的一个骰子掷出1到6这6种情况进行讨论
                    tmp[j + k - 1] += dp[j] * 1.0 / 6.0; //比较难的是这里的j+k-1
                }               
            }
            dp = tmp;
        }
        return dp;
    }
}
```

>  具体来看，由于新增骰子的点数只可能为 11 至 66 ，因此概率 f(n - 1, x)f(n−1,x) 仅与 f(n, x + 1)f(n,x+1) , f(n, x + 2)f(n,x+2), ... , f(n, x + 6)f(n,x+6) 相关。因而，遍历 f(n - 1)f(n−1) 中各点数和的概率，并将其相加至 f(n)f(n) 中所有相关项，即可完成 f(n - 1)f(n−1) 至 f(n)f(n) 的递推。
> 链接：https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/solution/jian-zhi-offer-60-n-ge-tou-zi-de-dian-sh-z36d/
>





#### 61 扑克牌中的顺子

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。

```
输入: [0,0,1,2,5]
输出: True
```

一开始有点懵，想着先排序，然后判断**0以外的数字最大减最小有没有大于4**。到这里就想不清楚了，后来才明白，再加上一个**非0数字不能重复**就够了，因为只要满足这样的条件，其他的地方都可以用0填补

```java
class Solution {
    public boolean isStraight(int[] nums) {
        Arrays.sort(nums);
        int i = 0;
        for(; i < 5; i++){
            if(nums[i] != 0) break;
        }
        if(i == 5){
            return true;
        }
        if(nums[4] - nums[i] > 4){
            return false;
        }
        i++;
        for(; i < 5; i++){
            if(nums[i] == nums[i - 1]){
                return false;
            }
        }
        return true;
    }
}
```

还可以用set，最大最小值可以在遍历的时候比较出来（以下来自题解）

```java
class Solution {
    public boolean isStraight(int[] nums) {
        Set<Integer> repeat = new HashSet<>();
        int max = 0, min = 14;
        for(int num : nums) {
            if(num == 0) continue; // 跳过大小王
            max = Math.max(max, num); // 最大牌
            min = Math.min(min, num); // 最小牌
            if(repeat.contains(num)) return false; // 若有重复，提前返回 false
            repeat.add(num); // 添加此牌至 Set
        }
        return max - min < 5; // 最大牌 - 最小牌 < 5 则可构成顺子
    }
}
```



#### 62 圆圈中最后剩下的数字（约瑟夫环）

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

之前的做法都是用boolean数组模拟，但是这样的复杂度是O(mn)看了题解发现可以用递归（效率较低），还可以把递归改成迭代（效率较高）

> 我们将上述问题建模为函数 f(n, m)，该函数的返回值为最终留下的元素的序号。
>
> 首先，长度为 n 的序列会先删除第 m % n 个元素，然后剩下一个长度为 n - 1 的序列。那么，我们可以递归地求解 f(n - 1, m)，就可以知道对于剩下的 n - 1 个元素，最终会留下第几个元素，记为x，则长度为 n 的序列最后一个删除的元素，应当是从 m % n 开始数的第 x 个元素。
>
> 因此有 f(n, m) = (m % n + x) % n = (m + x) % n

注释掉的是递归的写法，很简洁但是比较慢。非递归的写法要从最里层的递归开始循环到最外层

```java
class Solution {
    public int lastRemaining(int n, int m) {
        int res = 0;
        for(int i = 2; i <= n; i++){
            res = (res + m) % i;
        }
        return res;
        // if(n == 1) return 0;
        // return (m + lastRemaining(n - 1, m)) % n;
    }
}
```



#### 63 股票的最大利润（贪心）

**只进行一次交易**

只要记录前面的最小价格，将这个最小价格作为买入价格，然后将当前的价格作为售出价格，查看当前收益是不是最大收益。

```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) return 0;
        int res = 0;
        int min = prices[0];
        for(int i = 0; i < prices.length; i++){
            min = Math.min(prices[i], min);
            res = Math.max(prices[i] - min, res);
        }
        return res;
    }
}
```

还可以用滑动窗口（其他人的代码

一旦当前位置的值小于窗口的左边界，就移动窗口的左边界到当前的位置，右边界为下一个位置

```c++
int maxProfit(vector<int>& prices){
    int len = prices.size();
    if(len <= 1)
        return 0;
    
    int left = 0, right = 0, maxP = 0;
    while(right < len){
        if((prices[right]-prices[left]) < 0){
            left = right;
            right++;
            continue;
        }
        maxP = max(maxP, prices[right]-prices[left]);
        right++;
    }
    return maxP;
}
```

**另一种是可以多次交易，不能交叉**

当访问到一个 prices[i] 且 prices[i] - prices[i-1] > 0，那么就把 prices[i] - prices[i-1] 添加到收益中。



#### 64 求1+2+...+n（怪题）

要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）

看了题解做的，感觉这种题挺···让人无语的

第一种做法是用递归函数和逻辑运算符的短路性质

```java
class Solution {
    public int sumNums(int n) {
        boolean flag = (n > 0) && (n += sumNums(n - 1)) > 0;
        return n;
    }
}
```

第二种做法是用位运算实现乘法，类似于快速幂的思想但不太一样，快速幂是乘的，这里是加的

> 如果 B 的二进制表示下第 ii 位为 1，那么这一位对最后结果的贡献就是 A*(1<<i)A∗(1<<i) ，即 A<<iA<<i。我们遍历 B 二进制展开下的每一位，将所有贡献累加起来就是最后的答案，这个方法也被称作「俄罗斯农民乘法」
>

```c++
int quickMulti(int A, int B) {
    int ans = 0;
    for ( ; B; B >>= 1) {
        if (B & 1) {
            ans += A;//相当于ans += A * 1，因为只能是0或者1
        }
        A <<= 1;//相当于A = A * 2
    }
    return ans;
}
```

因为不能用循环语句，所以手动展开14层。。。无语

https://leetcode-cn.com/problems/qiu-12n-lcof/solution/qiu-12n-by-leetcode-solution/



#### 65 不用加减乘除做加法（位运算）

首先我看了题解

> 观察发现，**无进位和** 与 **异或运算** 规律相同，**进位** 和 **与运算** 规律相同（并需左移一位）
>
> （和 s ）=（非进位和 n ）+（进位 c ）
> s=a+b⇒s=n+c
>
> 循环求 n 和 c ，直至进位 c = 0；此时 s = n ，返回 n 即可。

一开始没懂，既然s=n+c为啥还要循环，直接做不就好了，后来才意识到不可以用加号，所以其实类似于递归，s=n+c=n'+c'=...当c==0时就可以跳出循环了

```java
class Solution {
    public int add(int a, int b) {
        int c, n;
        while(b != 0){    
            n = a ^ b;
            c = (a & b) << 1;//注意！这里要加小括号！不然会错
            a = n;
            b = c;
        }
        return a;
    }
}
```



#### 66 构建乘积数组（怪题，找规律）

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。**不能使用除法**。

```
输入: [1,2,3,4,5]
输出: [120,60,40,30,24]

所有元素乘积之和不会溢出 32 位整数
a.length <= 100000
```

毫无头绪，看了题解居然写出了和题解一模一样的代码。以及第一次提交的时候又又又忘记判断输入数组为空的情况了，气

这道题的关键在于把数组B写成一行一行的，就可以发现数组中每个元素之间的关系，**分成上三角和下三角求解**，下三角就是1, A[0], A[0]\*A[1], A[0]\*A[1]*A[2]...

上三角就是1, A[n], A[n]\*A[n-1], A[n]\*A[n-1]\*A[n-2]

```java
class Solution {
    public int[] constructArr(int[] a) {
        if(a.length == 0) return new int[0];
        int[] res = new int[a.length];//记得加！！！
        res[0] = 1;
        int cur = 1;
        for(int i = 1; i < a.length; i++){
            res[i] = res[i - 1] * a[i - 1]; 
        }
        for(int i = a.length - 2; i >= 0; i--){
            cur *= a[i + 1];
            res[i] *= cur; 
        }
        return res;
    }
}
```



#### 67 把字符串转换成整数

前后都可能有空格，有可能有正负号，可能越界，可能有无关字符

题目太长了。https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/

```java
class Solution {
    public int strToInt(String str) {    
        str = str.trim();
        if(str.length() == 0) return 0;
        boolean flag = true;
        if(str.charAt(0) == '+') str = str.substring(1);
        else if(str.charAt(0) == '-'){
            flag = false;
            str = str.substring(1);
        }

        if(str.length() == 0 || str.charAt(0) > '9' || str.charAt(0) < '0') return 0;
        long sum = 0;
        for(int i = 0; i < str.length(); i++){
            int cur = str.charAt(i) - '0';
            if(cur > 9 || cur < 0) break;
            sum = sum * 10 + cur;
            if(sum > 2147483647 && flag == true) return Integer.MAX_VALUE;
            if(sum > 2147483648L && flag == false) return Integer.MIN_VALUE; 
        }
        if(flag) return (int)sum;
        else return (int)(-sum);
    }
}
```

提交后出错是因为没有考虑到str为""或者"  "必须在首字符是不是正负号之前做判断

看了题解才意识到，题目的意思是不可以用long

> 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−2^31,  2^31 − 1]。如果数值超过这个范围，请返回  INT_MAX  或 INT_MIN 
>

```java
class Solution {
    public int strToInt(String str) {    
        str = str.trim();
        if(str.length() == 0) return 0;
        boolean flag = true;
        if(str.charAt(0) == '+') str = str.substring(1);
        else if(str.charAt(0) == '-'){
            flag = false;
            str = str.substring(1);
        }
        if(str.length() == 0 || str.charAt(0) > '9' || str.charAt(0) < '0') return 0;
        
        int sum = 0;
        for(int i = 0; i < str.length(); i++){
            int cur = str.charAt(i) - '0';
            if(cur > 9 || cur < 0) break;
            if(sum > 214748374 || (sum == 214748364 && cur > 7)) return flag ? Integer.MAX_VALUE : Integer.MIN_VALUE; 
            sum = sum * 10 + cur;
        }
        return flag ? sum : -sum;
    }
}
```

这个是参考题解写的，比较取巧的是如果是2147483648，不管正负都可以算成越界，因为如果是负的就返回-2147483648，还是一样



#### 68-1 二叉搜索树的最近公共祖先（做过好几次居然还是忘记怎么做）

一开始完全想错了，一直想怎么用递归。

其实要利用二叉搜索树的特性。可以先想比较简单的思路，分别求从root到p和root到q的路径，然后找分岔点，这个思路我能想到，但是嫌麻烦直接放弃了。

改进后的思路不需要存路径，只需要从根节点遍历下来，判断向左还是向右找。当p和q一个大于等于当前结点一个小于等于当前结点时就找到了

> 我们从根节点开始遍历；
>
> 如果当前节点的值大于 pp 和 qq 的值，说明 pp 和 qq 应该在当前节点的左子树，因此将当前节点移动到它的左子节点；
>
> 如果当前节点的值小于 pp 和 qq 的值，说明 pp 和 qq 应该在当前节点的右子树，因此将当前节点移动到它的右子节点；
>
> 如果当前节点的值不满足上述两条要求，那么说明当前节点就是「分岔点」
>

```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode iter = root;
        while(true){
            if(p.val > iter.val && q.val > iter.val) iter = iter.right;
            else if(p.val < iter.val && q.val < iter.val) iter = iter.left;
            else break;
        }
        return iter;        
    }
}
```



#### 68-2 二叉树的最近公共祖先（经典）

68-1一开始想用的就是这个递归的做法，想了好一会儿想出来了，递归函数返回值挺特别的

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

