### leetcode热门题 （1-100）

#### 1 两数之和（利用哈希表加速）

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** `target` 的那 **两个** 整数，并返回它们的数组下标。

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(map.containsKey(nums[i])){
                return new int[]{map.get(nums[i]), i};
            }
            map.put(target - nums[i], i);
        }
        return new int[]{0, 0};
    }
}
```

**注意** 循环外也得有返回值，返回new int[0]也行



#### 2 两数相加（链表合并模拟加法）

给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

我用了归并排序类似的做法，写得有点长

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
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode();
        ListNode pre = head;
        int tmp = 0, sum = 0;
        while(l1 != null && l2 != null){
            sum = (l1.val + l2.val + tmp) % 10;
            tmp = (l1.val + l2.val + tmp < 10) ? 0 : 1;
            ListNode curr = new ListNode(sum);
            pre.next = curr;
            pre = curr;
            l1 = l1.next;
            l2 = l2.next;
        } 
        while(l1 != null){
            sum = (l1.val + tmp) % 10;
            tmp = (l1.val + tmp < 10) ? 0 : 1;
            ListNode curr = new ListNode(sum);
            pre.next = curr;
            pre = curr;
            l1 = l1.next;
        }
        while(l2 != null){
            sum = (l2.val + tmp) % 10;
            tmp = (l2.val + tmp < 10) ? 0 : 1;
            ListNode curr = new ListNode(sum);
            pre.next = curr;
            pre = curr;
            l2 = l2.next;
        }
        if(tmp != 0){
            ListNode curr = new ListNode(1);
            pre.next = curr;
        }//第一次写的时候这里写成了while，结果造成了死循环
        return head.next;
    }
}
```

题解里把我的3个while合并成一个，如下

```java
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = null, tail = null;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int n1 = l1 != null ? l1.val : 0;
            int n2 = l2 != null ? l2.val : 0;
            int sum = n1 + n2 + carry;
            if (head == null) {
                head = tail = new ListNode(sum % 10);
            } else {
                //居然可以这么写，妙
                tail.next = new ListNode(sum % 10);
                tail = tail.next;
            }
            carry = sum / 10;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry > 0) {
            tail.next = new ListNode(carry);
        }
        return head;
    }
}
```



#### 3 无重复字符的最长子串（经典，双指针，滑动窗口）

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s.length() == 0) return 0;
        if(s.length() == 1) return 1;
        int max = 1;
        int left = 0, right = 0;
        HashSet<Character> set = new HashSet<>();
        while(right < s.length()){
            while(set.contains(s.charAt(right))){
                set.remove(s.charAt(left));
                left++;
            }
            set.add(s.charAt(right));
            max = Math.max(max, right - left + 1);
            right++;
        }
        return max;
    }
}
```

做过很多遍居然还是忘记怎么做了，只知道用set来判断是否重复，看了题解才知道是用双指针来做。。。哎



#### 4 寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

算法的时间复杂度应该为 O(log (m+n)) 。

示例 1：

```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
```

示例 2：

```
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

示例 3：

```
输入：nums1 = [], nums2 = [1]
输出：1.00000
```

最后用了最傻的做法，就是归并，意外的时间没有很多（打败100%）

就是空间还是有点浪费，其实应该可以不用整一个新的数组来存的，但写起来有点麻烦

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len = nums1.length + nums2.length;
        int pos1 = 0, pos2 = 0;
        int i = 0;
        int[] nums = new int[len / 2 + 1];
        while(pos1 < nums1.length && pos2 < nums2.length && i < nums.length){
            if(nums1[pos1] <= nums2[pos2]){
                nums[i] = nums1[pos1];
                pos1++;
            }
            else{
                nums[i] = nums2[pos2];
                pos2++;
            }
            i++;
        }
        while(i < nums.length && pos1 < nums1.length){
            nums[i] = nums1[pos1];
            pos1++;
            i++;
        }
        while(i < nums.length && pos2 < nums2.length){
            nums[i] = nums2[pos2];
            pos2++;
            i++;
        }
        return (nums[len / 2] + nums[(len - 1) / 2]) / 2.0; // 注意是2.0不是2
        // 这种写法可以不用分奇数和偶数讨论
    }
}
```

在题解里找了一下不用开新数组的写法，如下，可以理解成left和right双指针，right指针不停地找两个数组中下一个要merge进来的数，left保存right的前一个值

https://leetcode-cn.com/problems/median-of-two-sorted-arrays/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-2/

```java
public double findMedianSortedArrays(int[] A, int[] B) {
    int m = A.length;
    int n = B.length;
    int len = m + n;
    int left = -1, right = -1;
    int aStart = 0, bStart = 0;
    for (int i = 0; i <= len / 2; i++) {
        left = right;
        // 这个的条件语句很妙，还用到了短路
        if (aStart < m && (bStart >= n || A[aStart] < B[bStart])) {
            right = A[aStart++];
        } else {
            right = B[bStart++];
        }
    }
    if ((len & 1) == 0)
        return (left + right) / 2.0;
    else
        return right;
}
```

这个题解还提出了log(m+n)的解法，其实是求第k小的数的特殊解法

如下：

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int n = nums1.length;
    int m = nums2.length;
    int left = (n + m + 1) / 2;
    int right = (n + m + 2) / 2;
    //将偶数和奇数的情况合并，如果是奇数，会求两次同样的 k 。
    return (getKth(nums1, 0, n - 1, nums2, 0, m - 1, left) + getKth(nums1, 0, n - 1, nums2, 0, m - 1, right)) * 0.5;  
}
    
    private int getKth(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        //让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1 
        if (len1 > len2) return getKth(nums2, start2, end2, nums1, start1, end1, k);
        if (len1 == 0) return nums2[start2 + k - 1];

        if (k == 1) return Math.min(nums1[start1], nums2[start2]);

        int i = start1 + Math.min(len1, k / 2) - 1;
        int j = start2 + Math.min(len2, k / 2) - 1;

        if (nums1[i] > nums2[j]) {
            return getKth(nums1, start1, end1, nums2, j + 1, end2, k - (j - start2 + 1));
        }
        else {
            return getKth(nums1, i + 1, end1, nums2, start2, end2, k - (i - start1 + 1));
        }
    }
```

另一个思路相似的[题解](https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247484130&idx=5&sn=de027e77fd0cc185bd5d753bab38f5d0&chksm=fd9ca9fdcaeb20eb2d70190b3240d69ebc61c0d463c55ceb83eff76a4f624adcac00740faca7&token=583813353&lang=zh_CN&scene=21#wechat_redirect)的写法如下，比前一种更简洁，不过前一种更好理解

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int tot = nums1.length + nums2.length;
        if (tot % 2 == 0) {
            int left = find(nums1, 0, nums2, 0, tot / 2);
            int right = find(nums1, 0, nums2, 0, tot / 2 + 1);
            return (left + right) / 2.0;
        } else {
            return find(nums1, 0, nums2, 0, tot / 2 + 1);
        }
    }
    int find(int[] n1, int i, int[] n2, int j, int k) {
        if (n1.length - i > n2.length - j) return find(n2, j, n1, i, k);
        if (i >= n1.length) return n2[j + k - 1];
        if (k == 1) {
            return Math.min(n1[i], n2[j]);
        } else {
            int si = Math.min(i + (k / 2), n1.length), sj = j + k - (k / 2);
            if (n1[si - 1] > n2[sj - 1]) {
                return find(n1, i, n2, sj, k - (sj - j));
            } else {
                return find(n1, si, n2, j, k - (si - i));
            }
        }
    }
}
```



#### 5 最长回文子串

可以开二维布尔数组用dp，当s\[i] == s\[j]时，dp\[i]\[j] = dp\[i + 1]\[j - 1]

也可以从中心往两边扩展，以每个位置为中心向两边找最多可以对称几个，中心可以是一个点也可以是两个点，所以每个点都有两种情况，时间复杂度为O(n^2)

```java
class Solution {
    public String longestPalindrome(String s) {
        int left = 0, right = 0, max = 1;
        
        for(int i = 0; i < s.length(); i++){
            int tmp1 = longest(i, i, s);
            int tmp2 = longest(i, i + 1, s);
            int tmp = Math.max(tmp1, tmp2);
            if(tmp > max){
                left = i - (tmp - 1) / 2;
                right = i + tmp / 2;
                max = tmp;
            }    
        }
        return s.substring(left, right + 1);
    }
    public int longest(int left, int right, String s){
        while(left >= 0 && right < s.length()){
            if(s.charAt(left) == s.charAt(right)){    
                left--;
                right++;
            }
            else break;
        }
        return right - left - 1;
    }
}
```



#### 6 Z字形变换

将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

```
P   A   H   N
A P L S I I G
Y   I   R
```


之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

```java
class Solution {
    public String convert(String s, int numRows) {
        boolean flag = true;
        if(numRows == 1) return s;//！！！注意！！！
        StringBuilder[] strbs = new StringBuilder[numRows];
        for(int i = 0; i < numRows; i++){
            strbs[i] = new StringBuilder();
        }//必须初始化！！！否则报空指针
        int row = 0;
        for(int i = 0; i < s.length(); i++){
            strbs[row].append(s.charAt(i));
            if(row == numRows - 1){
                flag = false;
            }
            if(row == 0){
                flag = true;
            }
            if(flag){
                row++;
            }
            else{
                row--;
            }
        }
        StringBuilder res = new StringBuilder();
        for(int i = 0; i < numRows; i++){
            res.append(strbs[i]);
        }
        return res.toString();
    }
}
```

**注意：**numRows = 1, s = "AB" 这种情况需要特殊处理

 

#### 7 整数反转（错了两次的简单题）

```java
class Solution {
    public int reverse(int x) {
        boolean flag = true;
        if(x == -2147483648) return 0;
        if(x < 0){
            flag = false;
            x = -x;
        }
        int y = 0;
        while(x != 0){
            if(y > 214748364 || (y == 214748364 && x % 10 > 8) || (y == 214748364 && x % 10 == 8 && flag == true)){
                return 0;
            }
            y = y * 10 + x % 10;
            x = x / 10;
        }
        return flag ? y : -y;
    }
}
```

这题唯一的难点是**假设环境不允许存储 64 位整数（有符号或无符号）**，如果反转后整数超过 32 位的有符号整数的范围 `[−2^31, 2^31 − 1]` ，就返回 0。

所以需要在累加的时候加入一个判断

**注意：**比较特殊的输入为-2147483648，这个输入不能用x = -x这样的做法，会导致越界

经过观察后发现，负数可以和正数合并，而且不需要考虑y=214748364时下一位可能是8或者9，因为如果是的话，原来的数肯定是越界的，所以代码可以精简如下：

```java
class Solution {
    public int reverse(int x) {
        int y = 0;
        while(x != 0){
            if(y > 214748364 || y < -214748364){
                return 0;
            }
            y = y * 10 + x % 10;
            x = x / 10;
        }
        return y;
    }
}
```



#### 8 字符串转换整数

请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

函数 myAtoi(string s) 的算法如下：

读入字符串并丢弃无用的前导空格
检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
返回整数作为最终结果。
注意：

本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。

```java
class Solution {
    public int myAtoi(String s) {
        s = s.trim();
        if(s.length() == 0) return 0;//记得考虑s为空字符串的情况
        boolean flag = true;
        int cur_pos = 0;
        if(s.charAt(cur_pos) == '-'){
            flag = false;
            cur_pos++;
        }
        else if(s.charAt(cur_pos) == '+'){
            cur_pos++;
        } 
        
        int sum = 0;
        while(cur_pos < s.length()){
            if(s.charAt(cur_pos) > '9' || s.charAt(cur_pos) < '0'){
                break;
            }
            int cur = s.charAt(cur_pos) - '0';            
            if(sum > 214748364 || (sum == 214748364 && cur > 7)){
                return flag == true ? 2147483647 : -2147483648; 
            }
            sum = sum * 10 + cur;
            cur_pos++;           
        }
        return flag == true ? sum : -sum;//第一次提交的时候这里忘记判断flag了，记得判断！
    }
}
```

这道题挺简单的，只需要整数，但也要注意数组越界的问题和返回值要加正负



#### 9 回文数

这是一开始的想法

```java
class Solution {
    public boolean isPalindrome(int x) {
        if(x < 0) return false;
        List<Integer> list = new ArrayList<>();
        while(x != 0){
            list.add(x % 10);
            x /= 10;
        }
        int left = 0, right = list.size() - 1;
        while(left < right){
            if(list.get(left) != list.get(right)) return false;
            left++;
            right--;
        }
        return true;
    }
}
```

后来想到可以这么做，其实就是翻转数字加一个比较

```java
class Solution {
    public boolean isPalindrome(int x) {
        if(x < 0) return false;
        int sum = 0, y = x;
        while(x > 0){
            sum = sum * 10 + x % 10;
            x /= 10;
        }
        return sum == y;
    }
}
```

看了题解发现忘记考虑int越界的问题了，还侥幸过了

题解的思路是只翻转一半就进行比较，可以避免越界，重点在于判断到达一半了没

```java
class Solution {
    public boolean isPalindrome(int x) {
        if(x < 0) return false;
        if(x == 0) return true;
        if(x % 10 == 0) return false;//这三行很重要！
        int sum = 0;
        while(x > 0){
            sum = sum * 10 + x % 10;
            x /= 10;        
            if(sum >= x) break;
        }
        return sum == x || sum / 10 == x;
    }
}
```

这种写法写错了好几次，易错点是以0结尾但不是0的数字，应该作为特殊情况处理



#### 10 正则表达式匹配（二维DP）

给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

看了[官方题解](https://leetcode-cn.com/problems/regular-expression-matching/solution/zheng-ze-biao-da-shi-pi-pei-by-leetcode-solution/)做的，主要的思路：

字母 + 星号的组合在匹配的过程中，本质上只会有两种情况：

匹配 s 末尾的一个字符，将该字符扔掉，而该组合还可以继续进行匹配；

不匹配字符，将该组合扔掉，不再进行匹配。

我的代码如下：

```java
class Solution {
    public boolean isMatch(String s, String p) {
        s = "_" + s;
        p = "_" + p;
        int sLen = s.length();
        int pLen = p.length();
        boolean[][] dp = new boolean[pLen][sLen];
        dp[0][0] = true;
        for(int i = 1; i < pLen; i++){
            for(int j = 0; j < sLen; j++){
                if(match(s.charAt(j), p.charAt(i))){
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else if(p.charAt(i) == '*'){
                    if(match(s.charAt(j), p.charAt(i - 1))){
                        dp[i][j] = dp[i][j - 1] || dp[i - 2][j];
                    }
                    else{
                        dp[i][j] = dp[i - 2][j];
                    }
                    
                }
                else{
                    dp[i][j] = false;
                }
            }
        }
        return dp[pLen - 1][sLen - 1]; 
    }
    boolean match(char a, char b){
        return a == b || (b == '.' && a != '_');
    }
}
```

写的时候还是磕磕绊绊的，主要是下标加一减一的问题，还有处理空字符串的问题

我用的办法是在两个字符前都加了一个'_'用来表示空的状态，这样就可以不用处理加一减一的问题了，需要注意例如：a*这样的可以匹配空字符串，但是'.'不能匹配空字符串，所以在match函数里需要加一个判断

题解的写法如下：

```java
class Solution {
    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();

        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }
}
```

题解的做法则是把判断的部分放在了match函数里，其余部分就显得很好懂



#### 11 盛最多水的容器（没想到是双指针）

其实还有点贪心的感觉，一开始一直想单调栈之类的，后来看了题解才发现完全想错了

看懂怎么贪心之后就很好做了，一左一右两个指针，把比较小的那个往里移，不断计算并存储最大值就行了

```java
class Solution {
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int res = 0;
        while(left < right){
            res = Math.max(res, Math.min(height[left], height[right]) * (right - left));
            if(height[left] < height[right]){
                left++;
            }
            else{
                right--;
            }
        }
        return res;
    }
}
```



#### 12 整数转罗马数字（没想到是贪心）

我最开始的做法：

从高位到低位模拟，说实话**很傻**，如果数字范围再大点就不能这么搞了

```java
class Solution {
    public String intToRoman(int num) {
        StringBuilder strb = new StringBuilder();
        int cur = num / 1000;
        while(cur >= 1){
            strb.append("M");
            cur--;
        }
        num = num % 1000;
        cur = num / 100;
        if(cur == 9) strb.append("CM");
        else if(cur == 4) strb.append("CD");
        else if(cur >= 5){
            strb.append("D");
            while(cur > 5){
                strb.append("C");
                cur--;
            }
        }
        else if(cur > 0 && cur < 4){
            while(cur > 0){
                strb.append("C");
                cur--;
            }
        }
        num = num % 100;
        cur = num / 10;
        if(cur == 9) strb.append("XC");
        else if(cur == 4) strb.append("XL");
        else if(cur >= 5){
            strb.append("L");
            while(cur > 5){
                strb.append("X");
                cur--;
            }
        }
        else if(cur > 0 && cur < 4){
            while(cur > 0){
                strb.append("X");
                cur--;
            }
        }
        num = num % 10;
        cur = num;
        if(cur == 9) strb.append("IX");
        else if(cur == 4) strb.append("IV");
        else if(cur >= 5){
            strb.append("V");
            while(cur > 5){
                strb.append("I");
                cur--;
            }
        }
        else if(cur > 0 && cur < 4){
            while(cur > 0){
                strb.append("I");
                cur--;
            }
        }
        return strb.toString();
    }
}
```

这是我在leetcode里写过的最长的代码，用时和内存消耗倒是都击败了99，98

后来写了个比较短的版本，反而内存消耗比较大

```java
class Solution {
    public String intToRoman(int num) {
        StringBuilder strb = new StringBuilder();
        String[][] nums = {{"", "M", "MM", "MMM", "", "", "", "", "", ""},
                            {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"},
                            {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"},
                            {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}};
        int cur = 0;
        int divide = 1000;
        int i = 0;
        while(num != 0){
            cur = num / divide;    
            strb.append(nums[i][cur]);    
            num %= divide; 
            divide /= 10;
            i++;
        }
        return strb.toString();
    }
} 
```

最后看了题解，我还是太傻了

**硬编码**

就是上面第二种做法的改进版，直接计算出个十百千位，就是没想到时间空间复杂度居然会输那么多

```java
class Solution {
    public String intToRoman(int num) {
        StringBuilder strb = new StringBuilder();
        String[][] nums = {{"", "M", "MM", "MMM", "", "", "", "", "", ""},
                            {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"},
                            {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"},
                            {"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"}};
        
        return nums[0][num / 1000] + nums[1][num % 1000 / 100] + nums[2][num % 100 / 10] + nums[3][num % 10]; 
    }
} 
```

**贪心**

> 我们用来确定罗马数字的规则是：对于罗马数字从左到右的每一位，选择尽可能大的符号值。
>
> 根据罗马数字的唯一表示法，为了表示一个给定的整数num，我们寻找不超过num的最大符号值，将num减去该符号值，然后继续寻找不超过num的最大符号值，将该符号拼接在上一个找到的符号之后，循环直至num为0。最后得到的字符串即为num的罗马数字表示。
>
> 编程时，可以建立一个数值-符号对的列表，按数值从大到小排列。遍历列表中的每个数值-符号对，若当前数值value不超过num，则从num 中不断减去value，直至num小于value，然后遍历下一个数值-符号对。若遍历中num为0则跳出循环。
>

```java
class Solution {
    int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    String[] symbols = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

    public String intToRoman(int num) {
        StringBuffer roman = new StringBuffer();
        for(int i = 0; i < values.length; i++){

            while(num >= values[i]){
                roman.append(symbols[i]);
                num -= values[i];
            }
            if(num == 0) break;
        }
        return roman.toString();
    }
}
```



#### 13 罗马数字转整数

和上一题很相似，所以直接就用了相似的思路，贪心

```java
class Solution {
    int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
    String[] symbols = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
    public int romanToInt(String s) {
        int res = 0;
        for(int i = 0; i < symbols.length; i++){
            while(s.startsWith(symbols[i])){
                res += values[i];
                s = s.substring(symbols[i].length());
                if(s.length() == 0) return res;
            }
        }
        return res;
    }
}
```

结果题解居然搞模拟，一个字符一个字符地读，如果这个字符比后一个小，就减，否则就加，最后时间空间也没比我这个好



#### 14 最长公共前缀

编写一个函数来查找字符串数组中的最长公共前缀。

如果不存在公共前缀，返回空字符串 `""`。

浪费了大量时间在函数调用报错上，因为python写太多把小括号写成中括号了

还有一次报错是因为没有考虑到第一个字符为空字符串的情况导致的数组越界

```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        int pos = 0;
        while(true){
            if(pos >= strs[0].length()) break;
            for(int i = 1; i < strs.length; i++){
                if(pos >= strs[i].length() || strs[i].charAt(pos) != strs[0].charAt(pos)){
                    if(pos == 0) return "";
                    else return strs[0].substring(0, pos);
                }
            }
            pos++;
        }
        if(pos == 0) return "";
        return strs[0];
    }
}
```

做法有好几种，这是最容易想到的而且时间空间复杂度也比较低

```c++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if(strs.size() == 0){
            return "";
        }
        int len = strs[0].length();
        for(int i = 0; i < len; i++){
            char c = strs[0][i];
            for(int j = 1; j < strs.size(); j++){
                if(i >= strs[j].length() || strs[j][i] != c){
                    return strs[0].substr(0, i);
                }
            }
        }
        return strs[0];
    }
};
```



#### 15 三数之和（经典面试题，双指针变体）

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

看完题目没有思路，看了题解才明白是先利用排序避免重复，再利用双指针避免三重循环

但还是有各种判断和边界条件，需要考虑是否重复

所以自己写的第一版很混乱，加了很多条件语句

后来参考了题解写出了新的一版：

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++){
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            else {
                int k = nums.length - 1;
                for(int j = i + 1; j < k; j++){
                    if(j > i + 1 && nums[j] == nums[j - 1]) continue;
                    else{
                        while(k > j && nums[i] + nums[j] + nums[k] > 0){
                            k--;
                        }
                        if(j >= k) break;//这一步很重要！
                        if(nums[i] + nums[j] + nums[k] == 0){
                            List<Integer> newList = new ArrayList<>();
                            newList.add(nums[i]);
                            newList.add(nums[j]);
                            newList.add(nums[k]);
                            list.add(newList);
                        } 
                    }
                }
            }
        }
        return list;
    }   
}
```

这个做法是i和j都用更清晰的for循环，如果i和j都和之前不重复的话，k肯定也不重复，利用了这个特性减少了k的判断，代码更简洁，而使用for而不是while也减少了额外对j的判断和操作

在双指针遍历的时候，先固定左指针（外层循环），将右指针左移直到和不大于target（里层循环），然后判断

可以测试的边界例子有：[0, 0, 0], [-1, 0, 1], [0, 0, 1]等



#### 16 最接近的三数之和

给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。

输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。

我把前一题的代码改吧改吧写的下面这个：

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        int res = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for(int i = 0; i < nums.length; i++){
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            else {
                int k = nums.length - 1;
                for(int j = i + 1; j < k; j++){
                    if(j > i + 1 && nums[j] == nums[j - 1]) continue;
                    else{
                        while(k > j && nums[i] + nums[j] + nums[k] > target){
                            k--;
                        }
                        if(j >= k){//这里和前一题不一样，不一定会找到和小于target的，所以不能直接break
                            int sum = nums[i] + nums[j] + nums[j + 1];
                            if(Math.abs(sum - target) < Math.abs(res - target)) res = sum;
                            break;
                        }
                        int sum1 = nums[i] + nums[j] + nums[k];
                        
                        int sum2 = (k < nums.length - 1) ? nums[i] + nums[j] + nums[k + 1] : sum1;
                        if(Math.abs(sum1 - target) <= Math.abs(sum2 - target) && Math.abs(sum1 - target) < Math.abs(res - target)){
                            res = sum1;
                        } 
                        else if(Math.abs(sum2 - target) < Math.abs(sum1 - target) && Math.abs(sum2 - target) < Math.abs(res - target)){
                            res = sum2;
                        } 
                    }
                }
            }
        }
        return res;
    }
}
```

后来看了题解发现我又写麻烦了，题解的写法比较清晰，就是每次计算三个数的和，和target比较，看有没有比之前的结果更接近，决定左右指针要怎么移

```java
class Solution {
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int best = 10000000;

        for(int i = 0; i < nums.length; i++){
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            else{
                int j = i + 1, k = nums.length - 1;
                while(j < k){
                    int sum = nums[i] + nums[j] + nums[k];
                    if(sum == target) return target;
                    if(Math.abs(best - target) > Math.abs(sum - target)) best = sum;//这步很重要！
                    if(sum > target){
                        k--;
                        while(k > j && nums[k] == nums[k + 1]) k--;
                    }
                    else{
                        j++;
                        while(j < k && nums[j] == nums[j - 1]) j++;
                    }
                }
            }
        }
        return best;
    }
}
```

陷入了巨大的困惑，为什么相似的两道题题解的做法不一样而且都是更快，时间复杂度都是O(N^2)，但我写的就是更慢些，但其实也就差3ms



#### 17 电话号码的数字组合（backtracking经典）

给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

```
输入：digits = "23"
输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

```
输入：digits = ""
输出：[]
```



```java
class Solution {
    private static final String[] KEYS = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    public List<String> letterCombinations(String digits) {
        List<String> list = new ArrayList<>();
        if(digits.length() == 0) return list; //记得判空
        StringBuilder strb = new StringBuilder();
        search(list, strb, digits, 0);
        return list;
    }
    //其实这个pos参数可以用strb.length()代替
    public void search(List<String> list, StringBuilder strb, String digits, int pos){
        if(pos == digits.length()){
            list.add(strb.toString());
            return;
        }
        String cur = KEYS[digits.charAt(pos) - '0'];
        for(int i = 0; i < cur.length(); i++){
            strb.append(cur.charAt(i));
            search(list, strb, digits, pos + 1);
            strb.deleteCharAt(strb.length() - 1);
        }
        return;
    }
}
```



#### 18 四数之和

做法类似三数之和，时间复杂度O(N^3)，可以加入剪枝操作：

> 在确定第一个数之后，如果nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target，说明此时剩下的三个数无论取什么值，四数之和一定大于target，因此退出第一重循环；
> 在确定第一个数之后，如果nums[i] + nums[len - 3] + nums[len - 2] + nums[len - 1] < target，说明此时剩下的三个数无论取什么值，四数之和一定小于target，因此第一重循环直接进入下一轮，枚举nums[i+1]；
> 在确定前两个数之后，如果nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target，说明此时剩下的两个数无论取什么值，四数之和一定大于target，因此退出第二重循环；
> 在确定前两个数之后，如果nums[i] + nums[j] + nums[len - 2] + nums[len - 1] < target，说明此时剩下的两个数无论取什么值，四数之和一定小于 target，因此第二重循环直接进入下一轮，枚举nums[j+1]。



```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        int len = nums.length;
        if(len < 4) return res;
        Arrays.sort(nums);
        for(int i = 0; i < len - 3; i++){//这里是len - 3不是len，否则会数组越界
            if(i > 0 && nums[i] == nums[i - 1]) continue;
            if(nums[i] + nums[len - 3] + nums[len - 2] + nums[len - 1] < target) continue;
            if(nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
            for(int j = i + 1; j < len - 2; j++){//这里是len - 2不是len
                if(j > i + 1 && nums[j] == nums[j - 1]) continue;
                if(nums[i] + nums[j] + nums[len - 2] + nums[len - 1] < target) continue;
                if(nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) break;
                int right = len - 1;
                for(int left = j + 1; left < right; left++){
                    if(left > j + 1 && nums[left] == nums[left - 1]) continue;
                    while(left < right && nums[i] + nums[j] + nums[left] + nums[right] > target) right--;
                    //while循环里记得加left < right，否则下一个if语句里得写成left <= right
                    if(left == right) break;
                    if(nums[i] + nums[j] + nums[left] + nums[right] == target){
                        List<Integer> list = new ArrayList<>();
                        list.add(nums[i]);
                        list.add(nums[j]);
                        list.add(nums[left]);
                        list.add(nums[right]);
                        res.add(list);
                    }
                }
            }
        }
        return res;
    }
}
```



#### 19 删除链表的倒数第N个结点

经典题，用两个指针实现一次遍历实现，当第一个指针指到最后一个的时候第二个指针指向要删的结点的前一个结点

注意几种特殊情况，链表为空，**删第一个结点**，删最后一个结点

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
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode pre = head;
        ListNode cur = head;
        for(int i = 0; i < n; i++){
            pre = pre.next;
        }
        if(pre == null) return head.next;
        while(pre.next != null){
            pre = pre.next;
            cur = cur.next;
        }
        cur.next = cur.next.next;
        return head;
    }
}
```



#### 20 有效的括号

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合

```java
class Solution {
    Map<Character, Integer> map = new HashMap<>(){{
        put('(', 1);
        put(')', -1);
        put('{', 2);
        put('}', -2);
        put('[', 3);
        put(']', -3);
    }};
    public boolean isValid(String s) {
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < s.length(); i++){
            int tmp = map.get(s.charAt(i));
            if(tmp > 0){
                stack.push(tmp);
            }
            if(tmp < 0){
                // 注意判空
                if(stack.isEmpty() || stack.peek() + tmp != 0) return false;
                else stack.pop();                
            }
        }
        if(stack.isEmpty()) return true;
        return false;
    }
}
```



#### 21 合并两个有序链表（经典题）

将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

类似于归并排序，特点是不能开辟新空间，头脑混乱的时候会想不出来怎么做。其实代码很简单

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
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null) return l2;
        if(l2 == null) return l1;
        ListNode head = new ListNode();
        ListNode iter = head;
        while(l1 != null && l2 != null){
            if(l1.val <= l2.val){
                iter.next = l1;
                l1 = l1.next;// 要先保存新的l1
            }
            else{
                iter.next = l2;
                l2 = l2.next;
            }
            iter = iter.next;// 再保存新的iter
        }
        if(l1 == null){
            iter.next = l2;
        }
        else{
            iter.next = l1;
        }
        return head.next;    
    }
}
```

需要不断保存当前两个未处理的链表的头结点



#### 22 括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```
输入：n = 3
输出：["((()))","(()())","(())()","()(())","()()()"]
```

把问题想得太复杂了，看了题解才意识到其实是简单的**回溯**题

重点是：在加括号的时候保证每一步都是正确的，有两个规则

+ 当左括号的数量不超过规定数量时，可以加左括号
+ 当右括号的数量小于左括号时，可以加右括号

这样可以保证得到的最终结果一定是正确的组合，就不需要再验证一次是否有效

```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        generateParenthesis(current, 0, 0, n, result);
        return result;
    }
    public void generateParenthesis(StringBuilder current, int leftCount, int rightCount, int max, List<String> result){
        if (current.length() == max * 2){
            result.add(current.toString());
            return;
        }
        if (leftCount < max) {
            current.append("(");
            generateParenthesis(current, leftCount + 1, rightCount, max, result);
            current.deleteCharAt(current.length() - 1);
        }
        if (rightCount < leftCount) {
            current.append(")");
            generateParenthesis(current, leftCount, rightCount + 1, max, result);
            current.deleteCharAt(current.length() - 1);
        }
    }
}
```



#### 23 合并k个升序链表（归并：分治或者优先队列）

给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

这道题虽然是hard但其实很基础，只是合并两个升序链表的变体，如果能把合并两个写好，再掌握这道题要用的，例如优先队列，就没那么难

合并两个链表感觉很简单但也有些小坑

+ 定义一个虚头节点
+ 当前的指针和被选中的指针要记得不断向后移动
+ 记得判断在处理的两个节点有没有空

这道题用优先队列的写法：

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
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> queue = new PriorityQueue<ListNode>(new Comparator<ListNode>(){
            public int compare(ListNode o1, ListNode o2){
                return o1.val - o2.val;
            }
        });
        ListNode head = new ListNode();
        ListNode curr = head;
        for(int i = 0; i < lists.length; i++){
            if(lists[i] != null){
                queue.add(lists[i]);
            }
        }
        while(!queue.isEmpty()){
            ListNode next = queue.poll();
            curr.next = next;
            if(next.next != null){
                queue.add(next.next);
            }
            curr = next;            
        }
        return head.next;
    }
}
```

+ 优先队列的语法
+ queue poll一个出去之后，如果出去的那个节点还有next的话就把next加进去（不知道为啥写的时候死活不知道要这么写，还在纠结要怎么更新queue）
+ queue在add的时候要记得判空，不要把null加进queue里了
+ head要new，不然没有next属性



归并的写法：

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        int l = 0, r = lists.length - 1;
        return mergeKLists(lists, l, r);
    }

    public ListNode mergeKLists(ListNode[] lists, int l, int r){
        if(l == r) return lists[l];
        if(l > r) return null; // 处理lists为空的情况
        int mid = l + (r - l) / 2;
        return mergeTwoLists(mergeKLists(lists, l, mid), mergeKLists(lists, mid + 1, r));
    }

    public ListNode mergeTwoLists(ListNode left, ListNode right){
        ListNode head = new ListNode();
        ListNode curr = head;
        while(left != null && right != null){
            if(left.val < right.val){
                curr.next = left;
                left = left.next;
            }
            else{
                curr.next = right;
                right = right.next;
            }
            curr = curr.next;
        }
        curr.next = (left == null ? right : left);
        return head.next;
    }
}
```

分治归并比优先队列略好写点

两种写法都要考虑lists为空，lists[i]为空的情况



#### 24 两两交换链表中的节点

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

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
    public ListNode swapPairs(ListNode head) {
        ListNode preHead = new ListNode(0, head);
        ListNode pre = preHead;
        ListNode cur = head;
        ListNode next;
        while(cur != null){
            next = cur.next;
            if(next == null) break;
            pre.next = next;
            cur.next = next.next;
            next.next = cur;
            pre = cur;
            cur = pre.next;
        }
        return preHead.next;
    }
}
```

+ 比较麻烦的点是要处理[1]和[]和[1,2,3]这种情况，需要对cur和next指针进行判空



#### 25 k个一组翻转链表（经典题变种）

给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。

k 是一个正整数，它的值小于或等于链表的长度。

如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

进阶：

你可以设计一个只使用常数额外空间的算法来解决此问题吗？
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

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
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummyHead = new ListNode(0, head);
        ListNode pre = dummyHead;
        ListNode newTail = head;
        ListNode newHead = pre;
        while(newHead != null){
            for(int i = 0; i < k; i++){        
                newHead = newHead.next;//这两句的顺序很重要
                if(newHead == null) return dummyHead.next;
            }
            ListNode nextTail = newHead.next;
            reverse(newTail, newHead);
            newTail.next = nextTail;
            pre.next = newHead;
            pre = newTail;
            newHead = pre;
            newTail = pre.next;
        }
        
        return dummyHead.next;

    }
    public void reverse(ListNode newTail, ListNode newHead){
        ListNode pre = null;
        ListNode cur = newTail;
        ListNode next;
        while(pre != newHead){//!!!这里的条件必须用到newHead,如果用cur != null的话会把整个链表都翻转。这里只翻转从newTail到newHead的部分
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;     
        }
    }
}
```

+ 这道题其实不难，主要是感觉很复杂就一直不想做，其实就是进阶的翻转链表
+ 外层需要保存四个指针感觉比较繁琐，还需要判断如果这一段不到k个的话直接return
+ 里层的翻转链表需要保存三个指针，循环结束的条件需要注意



#### 26 删除有序数组中的重复项

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

代码很简单，主要思路是维护一个`nextPos`变量表示下一个不重复的元素应该放的位置。遍历数组，如果当前元素与前一个元素不相等，就放过去，然后更新`nextPos`

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int nextPos= 1;
        for(int i = 1; i < nums.length; i++) {
            if(nums[i] != nums[i - 1]) {
                nums[nextPos] = nums[i];
                nextPos++;
            }
        }
        return nextPos;
    }
}
```



#### 27 移除元素

给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。

不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。

元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。

和上一题几乎一样

```java
class Solution {
    public int removeElement(int[] nums, int val) {
        int nextPos= 0;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != val) {
                nums[nextPos] = nums[i];
                nextPos++;
            }
        }
        return nextPos;
    }
}
```



#### 28 实现strStr()（KMP算法，记）

实现 strStr() 函数。

给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

说明：

当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。

对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与 C 语言的 strstr() 以及 Java 的 indexOf() 定义相符。

https://leetcode-cn.com/problems/implement-strstr/solution/shua-chuan-lc-shuang-bai-po-su-jie-fa-km-tb86/

```java
class Solution {
    public int strStr(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        if (m == 0) {
            return 0;
        }
        int[] pi = new int[m];
        for (int i = 1, j = 0; i < m; i++) {
            while (j > 0 && needle.charAt(i) != needle.charAt(j)) {
                j = pi[j - 1];
            }
            if (needle.charAt(i) == needle.charAt(j)) {
                j++;
            }
            pi[i] = j;
        }
        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && haystack.charAt(i) != needle.charAt(j)) {
                j = pi[j - 1];
            }
            if (haystack.charAt(i) == needle.charAt(j)) {
                j++;
            }
            if (j == m) {
                return i - m + 1;
            }
        }
        return -1;

    }
}
```

时间复杂度O(M+N)



#### 29 两数相除（有点烦的题）

给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。

返回被除数 dividend 除以除数 divisor 得到的商。

整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2

提示：

被除数和除数均为 32 位有符号整数。
除数不为 0。
假设我们的环境只能存储 32 位有符号整数，其数值范围是 [−231,  231 − 1]。本题中，如果除法结果溢出，则返回 231 − 1。

因为不能用Long，又要考虑越界的问题，所以题解的做法是把正数转为负数，而不是通常的把负数转为正数。这里会比较容易出错

```java
class Solution {
    int MIN = Integer.MIN_VALUE, MAX = Integer.MAX_VALUE;
    int LIMIT = -1073741824; // MIN 的一半,用来防止越界
    // a是被除数, b是除数
    public int divide(int a, int b) {
        if (a == MIN && b == -1) return MAX;
        boolean flag = false;
        if ((a > 0 && b < 0) || (a < 0 && b > 0)) flag = true;
        if (a > 0) a = -a;
        if (b > 0) b = -b;
        int ans = 0;
        
        while (a <= b){
            int c = b, d = -1;
            // 不能写c + c >= a, 会越界
            // 这一步是找到最接近a的2的n次方乘b,也就是c, 再用a减去c
            // 这里的c >= limit和d >= limit不写也行
            while (c >= LIMIT && d >= LIMIT && c >= a - c){
                // c += c; d += d;
                // 用移位运算更快
                c = c << 1;
                d = d << 1;
            }
            a -= c;
            ans += d;
        }
        return flag ? ans : -ans;
    }
}
```



#### 30 串联所有单词的子串

给定一个字符串 s 和一些 长度相同 的单词 words 。找出 s 中恰好可以由 words 中所有单词串联形成的子串的起始位置。

注意子串要与 words 中的单词完全匹配，中间不能有其他字符 ，但不需要考虑 words 中单词串联的顺序。

示例 1：

```
输入：s = "barfoothefoobarman", words = ["foo","bar"]
输出：[0,9]
解释：
从索引 0 和 9 开始的子串分别是 "barfoo" 和 "foobar" 。
输出的顺序不重要, [9,0] 也是有效答案。
```

只要知道是用哈希表来做其实这道题就不难了，可惜我是看了题解才知道的

首先拿一个哈希表存words数组，key是string，value是次数，利用哈希表就可以对不考虑顺序的集合进行比较，不用set是因为words数组里面可能有单词出现了不止一次

接着遍历string，取出子串判断是否符合条件，我使用的策略是先判断一下这个子串的第一个单词是否符合要求，符合的话再取，然后就是为这个子串建立第二个哈希表，当第二个哈希表中某个单词出现的次数大于第一个哈希表时就break，还有当这个单词在第一个哈希表中不存在时break，就不需要其他条件了，像是次数不够的情况必然出现前两种之一。

```java
class Solution {
    public List<Integer> findSubstring(String s, String[] words) {
        Map<String, Integer> map1 = new HashMap<>();
        for(int i = 0; i < words.length; i++){
            map1.put(words[i], map1.getOrDefault(words[i], 0) + 1);
        }
        int len = words[0].length();
        int totalLen = len * words.length;
        List<Integer> res = new ArrayList<>();
        for(int i = 0; i <= s.length() - totalLen; i++){
            if(map1.containsKey(s.substring(i, i + len))){
                String cur = s.substring(i, i + totalLen);
                Map<String, Integer> map2 = new HashMap<>();
                int j;
                for(j = 0; j < totalLen; j = j + len){
                    String curSub = cur.substring(j, j + len);
                    if(map1.containsKey(curSub)){
                        map2.put(curSub, map2.getOrDefault(curSub, 0) + 1);
                        if(map1.get(curSub) < map2.get(curSub)) break;
                    }
                    else break;
                }
                if(j == totalLen) res.add(i);
            }
        }
        return res;
    }
}
```



#### 31 下一个排列（数学推导）

实现获取 下一个排列 的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列（即，组合出下一个更大的整数）。

如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

必须 原地 修改，只允许使用额外常数空间。

**示例 1：**

```
输入：nums = [1,2,3]
输出：[1,3,2]
```

**示例 2：**

```
输入：nums = [3,2,1]
输出：[1,2,3]
```

这道题也是看了题解才知道怎么做的

+ 要把数字变大相当于要把其中的一位数字变大
+ 这位数字应该要尽可能靠右
+ 这个数字变大的程度要尽可能小
+ 这个数字变大之后，它后面的数字组合应该尽可能小

所以我们要找的是从右往左第一个小于右边的数的数字，将它和它右侧比它大的最小的数字交换位置，然后将它后面的数字从小到大排

**注意** 这个被选中的数字的右侧其实是一个递减序列，交换过位置之后依然是递减序列（可以证明），所以排序只需要颠倒一下就行

```java
class Solution {
    public void nextPermutation(int[] nums) {
        int i = 0;
        for(i = nums.length - 2; i >= 0; i--){
            if(nums[i] < nums[i + 1]){
                for(int j = nums.length - 1; j > i; j--){
                    if(nums[j] > nums[i]){
                        int tmp = nums[j];
                        nums[j] = nums[i];
                        nums[i] = tmp;
                        break;
                    }
                }
                break;
            }
        }
        int left = i + 1;
        int right = nums.length - 1;
        while(left < right){
            int tmp = nums[left];
            nums[left] = nums[right];
            nums[right] = tmp;
            left++;
            right--;
        }
    }
}
```



#### 32 最长有效括号（动态规划经典变体）

给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。

示例 1：

```
输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
```

示例 2：

```
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
```

示例 3：

```
输入：s = ""
输出：0
```

首先需要想到动规。

然后需要将dp[i]表示为以i结尾的字符串最长有效的长度

+ 如果i的位置是左括号，肯定无效，直接填0
+ 如果i的位置的右括号，看它左侧是什么。如果是左括号，正好凑成一对，则dp[i]=dp[i-2]+2。如果是右括号，需要跨过dp[i-1]找到i-dp[i-1]-1位置的字符，看是否是左括号，如果是的话就能配对，dp[i]=2+dp[i-1]+dp[i-dp[i-1]-2]
+ 需要考虑一些边界情况

```java
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        int[] dp = new int[len];
        int res = 0;
        for(int i = 0; i < len; i++){
            if(s.charAt(i) == '('){
                dp[i] = 0;
            }
            else if(i > 0){
                if(s.charAt(i - 1) == '('){
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                }
                else if(i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '('){
                    dp[i] = dp[i - 1] + 2;
                    if (i - dp[i - 1] - 1 >= 1){
                        dp[i] += dp[i - dp[i - 1] - 2];
                    }
                }
                res = Math.max(res, dp[i]);
            }
        }
        return res;
    }
}
```



#### 33 搜索旋转排序数组（二分查找经典变体，81题是加强版）

整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

示例 1：

```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

示例 2：

```
输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1
```

**注意**！！！

一个很容易错的测试用例是

```
输入：nums = [3,1], target = 1
输出：1
```

看了题解的思路写的，重点在于判断mid的左右两边哪一边是有序的，利用有序的那一边来判断target有没有在那部分，然后修改搜索范围

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right){ 
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) return mid;
            // 左侧有序
            if(nums[left] <= nums[mid]){ //注意这里要用小于等于，用来处理left = mid的情况
                if(target >= nums[left] && target < nums[mid]){ //这里用小于mid而不是小于等于mid-1是为了避免数组越界
                    right = mid - 1;
                }
                else{
                    left = mid + 1;
                }
            }
            // 右侧有序
            else{
                if(target > nums[mid] && target <= nums[right]){ //这里用大于mid而不是大于等于mid+1是为了避免数组越界
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

这道题比较纠结的点是mid到底要算在左边还是右边，我想了很久感觉应该是都算



#### 34 在排序数组中查找元素的第一个和最后一个位置（二分查找经典变体）

给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

进阶：

你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？


示例 1：

```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
```

示例 2：

```
输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

示例 3：

```
输入：nums = [], target = 0
输出：[-1,-1]
```

找第一个和最后一个相当于两道题，找第一个比较简单一点。另外需要注意数组越界的判断(对应找不到这个数的情况)

主要的点就是当nums[mid] == target的时候，left或者right要怎么变，以及循环终止条件怎么定，还有怎么避免死循环，多举几个例子，比如：

[8,8,8], target = 8

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        if(nums.length == 0) return new int[]{-1, -1};
        int left = 0, right = nums.length - 1;
        int resLeft = 0, resRight = 0;
        while(left < right){
            int mid = left + (right - left) / 2;
            if(nums[mid] > target) right = mid - 1;
            else if(nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        if(right >= 0 && right < nums.length && nums[right] == target) resLeft = right;
        else resLeft = -1;
        left = 0;
        right = nums.length - 1;
        while(left < right){
            int mid = left + (right - left) / 2 + 1; // 这里是避免死循环的重点，和前一个做法有点不同，是为了让left能不断向右移动
            if(nums[mid] > target) right = mid - 1;
            else if(nums[mid] < target) left = mid + 1;
            else {
                left = mid;
            }
        }
        if(left >= 0 && left < nums.length && nums[left] == target) resRight = left;
        else resRight = -1;
        return new int[]{resLeft, resRight};
    }
}
```



#### 35 搜索插入位置（二分查找本体）

给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(nums[mid] == target) return mid;
            else if(nums[mid] < target) left = mid + 1;
            else right = mid - 1;
        }
        return left;
    }
}
```



#### 36 有效的数独

请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。

+ 数字 1-9 在每一行只能出现一次。
+ 数字 1-9 在每一列只能出现一次。
+ 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）


注意：

+ 一个有效的数独（部分已被填充）不一定是可解的。
+ 只需要根据以上规则，验证已经填入的数字是否有效即可。
+ 空白格用 '.' 表示。

思路其实很简单，一开始想的是用set，代码如下

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        List<HashSet<Character>> setsColumn = new ArrayList<>(9);
        List<HashSet<Character>> setsRow = new ArrayList<>(9);
        List<HashSet<Character>> setsBlock = new ArrayList<>(9);
        init(setsColumn);
        init(setsRow);
        init(setsBlock);
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                char curr = board[i][j];
                if(curr != '.'){
                    if(setsColumn.get(j).contains(curr)) return false;
                    else setsColumn.get(j).add(curr);
                    if(setsRow.get(i).contains(curr)) return false;
                    else setsRow.get(i).add(curr);
                    int blockNum = (i / 3) * 3 + (j / 3);
                    if(setsBlock.get(blockNum).contains(curr)) return false;
                    else setsBlock.get(blockNum).add(curr);
                }
            }
        }
        return true;
    }
    void init(List<HashSet<Character>> list){ // 记得初始化，不然会在get的时候报错out of bound
        for(int i = 0; i < 9; i++){ // 注意！这里要用9，不能用list.size()，因为size还是0，这个Bug愣是找了半天
            list.add(new HashSet<Character>());
        }
    }
}
```

**set的速度有点慢，后来改用boolean数组**，参考了一下题解的写法，变快了很多（其实就差了1ms，但是打败了100%）

```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        boolean[][] columns = new boolean[9][9];
        boolean[][] rows = new boolean[9][9];
        boolean[][] blocks = new boolean[9][9];
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                char curr = board[i][j];
                if(curr != '.'){
                    int index = curr - '0' - 1;
                    int blockNum = (i / 3) * 3 + (j / 3);
                    if(columns[j][index] || rows[i][index] || blocks[blockNum][index]) return false;
                    else columns[j][index] = rows[i][index] = blocks[blockNum][index] = true;
                }
            }
        }
        return true;
    }
}
```



#### 37 解数独（回溯）

算是前一题的升级版，区别是需要解出来，在原来的二维数组上修改，把答案填进去，用经典的回溯做就行，看了题解才发现思路不难

+ **用一个list来存放空着的位置**，这里是按照先行再列的顺序遍历的
+ 用和前一题同样的三个数组存放行，列，块的数字出现情况，用于后续判断是否为有效数独
+ 回溯的时候，在递归调用前要记得将这三个数组的状态设为true，调用后设为false

```java
class Solution {
    boolean[][] rows = new boolean[9][9];
    boolean[][] columns = new boolean[9][9];
    boolean[][] blocks = new boolean[9][9];
    List<int[]> spaces = new ArrayList<>();
    //这部分设置成类变量可以减少传参
    
    public void solveSudoku(char[][] board) {  
        for(int i = 0; i < 9; i++){
            for(int j = 0; j < 9; j++){
                if(board[i][j] == '.'){
                    int[] space = new int[]{i, j};
                    spaces.add(space);
                }
                else{
                    int index = board[i][j] - '0' - 1;
                    int blockNum = (i / 3) * 3 + (j / 3);
                    rows[i][index] = columns[j][index] = blocks[blockNum][index] = true;
                }
            }
        }
        solveSudoku(board, 0);
    }
    private boolean solveSudoku(char[][] board, int pos){
        if(pos == spaces.size()) return true;
        int[] space = spaces.get(pos);
        int i = space[0];
        int j = space[1];
        int blockNum = (i / 3) * 3 + (j / 3);
        for(int digit = 0; digit < 9; digit++){
            if(!rows[i][digit] && !columns[j][digit] && !blocks[blockNum][digit]){
                rows[i][digit] = columns[j][digit] = blocks[blockNum][digit] = true;
                board[i][j] = (char)('0' + digit + 1);
                if(solveSudoku(board, pos + 1)) return true; // 如果有解就提前返回
                board[i][j] = '.';
                rows[i][digit] = columns[j][digit] = blocks[blockNum][digit] = false;
            }
        }
        return false;
    }
}
```



#### 38 外观数列

给定一个正整数 n ，输出外观数列的第 n 项。

「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。

你可以将其视作是由递归公式定义的数字字符串序列：

countAndSay(1) = "1"
countAndSay(n) 是对 countAndSay(n-1) 的描述，然后转换成另一个数字字符串。
前五项如下：

```
1.     1
2.     11
3.     21
4.     1211
5.     111221
       第一项是数字 1 
       描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
       描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
       描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
       描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
```

简单的递归，没什么稀奇的，注意一下状态变化的点，以及最后循环结束后的处理

```java
class Solution {
    public String countAndSay(int n) {
        if(n == 1) return "1";
        String pre = countAndSay(n - 1);
        StringBuilder res = new StringBuilder();
        int count = 1;
        char cur = pre.charAt(0);
        for(int i = 1; i < pre.length(); i++){
            if(pre.charAt(i) == pre.charAt(i - 1)){
                count++;
            }
            else{
                res.append(String.valueOf(count));
                res.append(cur);
                cur = pre.charAt(i);
                count = 1;
            }
        }
        res.append(String.valueOf(count));
        res.append(cur);
        return res.toString();
    }
}
```



#### 39 组合总和（又是回溯，注意copy）

给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

示例 1：

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
```

示例 2：

```
输入: candidates = [2,3,5], target = 8
输出: [[2,2,2,2],[2,3,3],[3,5]]
```

示例 3：

```
输入: candidates = [2], target = 1
输出: []
```

示例 4：

```
输入: candidates = [1], target = 1
输出: [[1]]
```

示例 5：

```
输入: candidates = [1], target = 2
输出: [[1,1]]
```

这道题比较需要注意的是

+ 组合不能重复。我采用的策略是从左往右选择元素，不能往回找，比如[2,3,5]当我把3放进去的时候，后面的选择就只能选3或者5，不能选2了
+ 注意list的copy

```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        dfs(candidates, 0, target, res, list);
        return res;
    }
    void dfs(int[] candidates, int pos, int target, List<List<Integer>> res, List<Integer> list){
        if(target == 0){    
            res.add(new ArrayList<>(list));//注意需要拷贝
            return;
        }
        for(int i = pos; i < candidates.length; i++){
            if(candidates[i] <= target){
                list.add(candidates[i]);
                dfs(candidates, i, target - candidates[i], res, list);
                list.remove(list.size() - 1);
            }
        }
        return;
    }
}
```



#### 40 组合总和二（比上一题难了不少）

给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用一次。

注意：解集不能包含重复的组合。 

示例 1:

```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
输出:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
```

大部分代码可以用上一题的，比较难的点是要保证解里面的组合不重复，试过了用set来去重，会超出时间限制，所以得靠自己制定规则

我用的方法是

+ 先把备选数组排序
+ 构造一个list用来存放已经选择了哪些位置（后来改成用一个boolean数组来存visited状态，速度提升了不少）
+ 在选择数字之前先判断它和前一个位置的数字是否相等，如果相等且前一个数字没有被visited，那这个数字也不要选（这里有点难懂，建议举个例子看看）
+ 举例[1,1,1,1,1] target=3

```java
class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        boolean[] visited = new boolean[candidates.length];
        dfs(candidates, 0, target, res, list, visited);
        return res;
    }
    void dfs(int[] candidates, int pos, int target, List<List<Integer>> res, List<Integer> list, boolean[] visited){
        if(target == 0){    
            res.add(new ArrayList<>(list));
            return;
        }
        
        for(int i = pos; i < candidates.length; i++){
            if(i > 0 && candidates[i] == candidates[i - 1] && !visited[i - 1]) {
                continue;
            }
            if(candidates[i] <= target){
                list.add(candidates[i]);
                visited[i] = true;
                dfs(candidates, i + 1, target - candidates[i], res, list, visited);
                list.remove(list.size() - 1);
                visited[i] = false;
            }
            else break;
        }
        return;
    }
}
```

90题用的是preVisited变量，这道题不能这样做，必须得用数组，是因为用pre的话只能记录前一个位置有没有，而这里是可以跳着的。文字解释起来不太好懂，画个树状图就很清楚了



#### 41 缺失的第一个正数（Hard, 桶排序, 因为swap找了半天bug）

给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。 

进阶：你可以实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案吗？

从[宫水三叶的刷题日记](https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247486339&idx=1&sn=9b351c45a2fe1666ea2905dccbc11d0c&chksm=fd9ca09ccaeb298a4dfcc071911e9e99594befdc5f62860e7b602d11fc862d6bddc43f704ecc&token=1087019265&lang=zh_CN&scene=21#wechat_redirect)看的两种思路

第一种是先排序，然后查找1到n的数哪一个不在数组中。思路简单，复杂度为nlogn，由于n<=300，log300<10可以视为常数

第二种是桶排序，把每个数放在它应该出现的位置上（看不太懂为啥叫做桶排序，感觉不太像）

最气的是在交换数组中两个位置的值的时候，第一种和第三种做法可以，第二种不行，看了很久很久才被提醒说是nums[i]在被赋值之后就不能再拿来用了

```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        // Arrays.sort(nums);
        // for(int i = 1; i <= nums.length; i++){
        //     if(Arrays.binarySearch(nums, i) < 0) return i;
        // }
        // return nums.length + 1;
        int n = nums.length;
        for(int i = 0; i < n; i++){
            while(nums[i] >= 1 && nums[i] <= n && nums[i] != i + 1 && nums[i] != nums[nums[i] - 1]){
                // 这个可以
                int tmp = nums[nums[i] - 1];
                nums[nums[i] - 1] = nums[i];
                nums[i] = tmp;
                
                // 这个不行
                // int tmp = nums[i];
                // nums[i] = nums[nums[i] - 1];//nums[i]被修改了！
                // nums[nums[i] - 1] = tmp;//这里不能再用nums[i]了！！
                
                // 这个可以
                //swap(nums, i, nums[i] - 1);//传参传的值，不影响
            }
        }
        for(int i = 0; i < n; i++){
            if(nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }
    void swap(int[] nums, int a, int b) {
        int c = nums[a];
        nums[a] = nums[b];
        nums[b] = c;
    }
}
```



#### 42 接雨水（超级经典题，多解法）

https://leetcode-cn.com/problems/trapping-rain-water/solution/jie-yu-shui-by-leetcode/

这个官方题解就很不错

参考题解写的单调栈（递减）的代码如下：

```java
class Solution {
    public int trap(int[] height) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        for(int i = 0; i < height.length; i++){
            while(!stack.isEmpty() && height[i] > height[stack.peek()]){
                int top = stack.pop();
                if(stack.isEmpty()) break; // 注意判空
                int width = i - stack.peek() - 1;
                int height_ = Math.min(height[i], height[stack.peek()]) - height[top];
                res += width * height_;
            }
            stack.push(i);
        }
        return res;
    }
}
```

另一种做法，题解说DP但我感觉只是记忆化，代码如下。不过这个写法更简单，推荐用这个，缺点是可能不太适用其他题吧

```java
class Solution {
    public int trap(int[] height) {
        int len = height.length;
        if(len == 0) return 0;
        int res = 0;
        int[] leftmax = new int[len];
        int[] rightmax = new int[len];
        int left = 0, right = 0;
        for(int i = 0; i < len; i++){
            left = Math.max(left, height[i]);
            leftmax[i] = left;
        }
        for(int i = len - 1; i >= 0; i--){
            right = Math.max(right, height[i]);
            rightmax[i] = right;
        }
        for(int i = 0; i < len; i++){
            res += Math.min(leftmax[i], rightmax[i]) - height[i];
        }
        return res;
    }
}
```

最后还有一种是双指针，使用一次遍历完成，没太绕明白



#### 43 字符串相乘（模拟加法和乘法）

给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。

示例 1:

```
输入: num1 = "2", num2 = "3"
输出: "6"
```

示例 2:

```
输入: num1 = "123", num2 = "456"
输出: "56088"
```

说明：

```
num1 和 num2 的长度小于110。
num1 和 num2 只包含数字 0-9。
num1 和 num2 均不以零开头，除非是数字 0 本身。
不能使用任何标准库的大数类型（比如 BigInteger）或直接将输入转换为整数来处理
```

这道题涉及了加法和乘法，乘法是循环乘数的每一位去乘以被乘数

```java
class Solution {
    public String multiply(String num1, String num2) {
        if(num1.equals("0") || num2.equals("0")) return "0"; // 记得判断0的情况
        int len1 = num1.length();
        int len2 = num2.length();
        StringBuilder res = new StringBuilder();
        for(int i = len2 - 1; i >= 0; i--){
            StringBuilder cur = new StringBuilder();
            // 这里补零，很重要
            for(int k = len2 - 1; k > i; k--){
                cur.append('0');
            }
            int add = 0;
            int digit2 = num2.charAt(i) - '0';
            for(int j = len1 - 1; j >= 0; j--){
                int digit1 = num1.charAt(j) - '0';
                int multiply = digit1 * digit2 + add;
                add = multiply / 10;
                cur.append((char)(multiply % 10 + '0'));
            }
            if(add > 0) cur.append((char)(add % 10 + '0'));
            res = add(res.toString(), cur.reverse().toString());
        }
        return res.toString();

    }
    StringBuilder add(String num1, String num2){
        StringBuilder res = new StringBuilder();
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        int add = 0;
        while(i >= 0 || j >= 0 || add != 0){ // 这么写可以避免分类讨论
            int digit1 = (i >= 0) ? (num1.charAt(i) - '0') : 0;
            int digit2 = (j >= 0) ? (num2.charAt(j) - '0') : 0;
            int sum = digit1 + digit2 + add;
            add = sum / 10;
            res.append((char)(sum % 10 + '0'));
            i--;
            j--;
        }
        return res.reverse();
    }
}
```

这个做法的耗时比较多，主要集中在字符串相加的过程，看了[题解](https://leetcode-cn.com/problems/multiply-strings/solution/you-hua-ban-shu-shi-da-bai-994-by-breezean/)的另一种做法，用一个int数组来存放计算的每一位结果，需要用到几个规律（**重点**）

+ m位数乘以n位数的结果位数是m+n或者m+n-1
+ num1[i] * num2[j] 的结果最多两位，第一位在res[i + j]，第二位在res[i + j + 1]

```java
class Solution {
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int[] res = new int[num1.length() + num2.length()];
        for (int i = num1.length() - 1; i >= 0; i--) {
            int n1 = num1.charAt(i) - '0';
            for (int j = num2.length() - 1; j >= 0; j--) {
                int n2 = num2.charAt(j) - '0';
                int sum = (res[i + j + 1] + n1 * n2); // res[i+j+1]上可能有之前的进位
                res[i + j + 1] = sum % 10;
                res[i + j] += sum / 10;// 更新进位，注意这里是+=，不是=
            }
        }

        StringBuilder result = new StringBuilder();
        for (int i = 0; i < res.length; i++) {
            if (i == 0 && res[i] == 0) continue;
            result.append(res[i]);// 查了语法，可以这么写，也可以(char)(res[i] + '0')
        }
        return result.toString();
    }
}
```

上面这种做法巧妙地避免了再进行字符串加法的工作，代码也短了很多，速度快了不少。这么一想第一种做法也可以稍微改下比如用List\<Integer>来存每次的结果



#### 44 通配符匹配（比第10题略简单）

给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。

'?' 可以匹配任何单个字符。
'*' 可以匹配任意字符串（包括空字符串）。
两个字符串完全匹配才算匹配成功。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。

只要会了第10题，这道题就很好做了

```java
class Solution {
    public boolean isMatch(String s, String p) {
        s = '_' + s;
        p = '_' + p;
        int sLen = s.length();
        int pLen = p.length();
        boolean[][] dp = new boolean[pLen][sLen];
        dp[0][0] = true;
        for(int i = 1; i < pLen; i++){
            for(int j = 0; j < sLen; j++){
                if(p.charAt(i) == '*'){
                    if(j > 0){
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                    }
                    else{
                        dp[i][j] = dp[i - 1][j];
                    }    
                }
                else if(match(s.charAt(j), p.charAt(i))){
                    dp[i][j] = dp[i - 1][j - 1];
                }
            }
        }
        return dp[pLen - 1][sLen - 1];
    }
    boolean match(char a, char b){
        return a == b || (b == '?' && a != '_') || (b == '*');
    }
}
```



#### 45 跳跃游戏二（有点难的贪心）

给你一个非负整数数组 nums ，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

假设你总是可以到达数组的最后一个位置。 

示例 1:

```
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

我一开始用的是傻傻的两重循环DP

```java
class Solution {
    public int jump(int[] nums) {
        int[] dp = new int[nums.length];
        for(int i = 1; i < nums.length; i++){
            dp[i] = 10000;
            for(int j = 0; j < i; j++){
                if(nums[j] + j >= i){
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[nums.length - 1];
    }
}
```

时间特别久

后来看了题解的贪心，只需要一重循环。遍历数组，求出每次能到达的最远的位置（变量max）

维护当前能够到达的最大下标位置，记为边界（变量end）。我们从左到右遍历数组，到达边界时，更新边界并将跳跃次数增加 1。

> 在遍历数组时，我们不访问最后一个元素，这是因为在访问最后一个元素之前，我们的边界一定大于等于最后一个位置，否则就无法跳到最后一个位置了。如果访问最后一个元素，在边界正好为最后一个位置的情况下，我们会增加一次「不必要的跳跃次数」，因此我们不必访问最后一个元素。
> 链接：https://leetcode-cn.com/problems/jump-game-ii/solution/tiao-yue-you-xi-ii-by-leetcode-solution/

```java
class Solution {
    public int jump(int[] nums) {
        int len = nums.length;
        int max = 0;
        int end = 0;
        int step = 0;
        int[] dp = new int[len];
        for(int i = 0; i < len - 1; i++){
            max = Math.max(max, nums[i] + i);
            if(i == end){
                end = max;
                step++;
            }
        }
        return step;
    }
}
```



#### 46 全排列（老回溯题了）

给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

示例 1：

```
输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

示例 2：

```
输入：nums = [0,1]
输出：[[0,1],[1,0]]
```

示例 3：

```
输入：nums = [1]
输出：[[1]]
```


提示：

+ 1 <= nums.length <= 6
+ -10 <= nums[i] <= 10
+ nums 中的所有整数 互不相同

```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        dfs(res, list, visited, nums);
        return res;
    }
    void dfs(List<List<Integer>> res, List<Integer> list, boolean[] visited, int[] nums){
        if(list.size() == nums.length){
            res.add(new ArrayList<>(list));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(!visited[i]){
                visited[i] = true;
                list.add(nums[i]);
                dfs(res, list, visited, nums);
                list.remove(list.size() - 1);
                visited[i] = false;
            }
        }
    }
}
```



#### 47 全排列二（类似第40题）

给定一个可包含重复数字的序列 `nums` ，**按任意顺序** 返回所有不重复的全排列。

和全排列一不同在数组中包含重复的数字

**示例 1：**

```
输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]
```

+ 要记得先给数组排序！

```java
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        Arrays.sort(nums);// 注意！
        dfs(res, list, visited, nums);
        return res;
    }
    void dfs(List<List<Integer>> res, List<Integer> list, boolean[] visited, int[] nums){
        if(list.size() == nums.length){
            res.add(new ArrayList<>(list));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(!visited[i]){
                if(i > 0 && nums[i] == nums[i - 1] && !visited[i - 1]) continue;
                visited[i] = true;
                list.add(nums[i]);
                dfs(res, list, visited, nums);
                list.remove(list.size() - 1);
                visited[i] = false;
            }
        }
    }
}
```

用来保证list不重复的方法是：如果一个数字和前一个的值相同，就优先选前面的，也就是说，如果前面的数字visited为false，且值和当前的数字相同，那就不要选当前的数字



#### 48 旋转图像（有点巧妙的找规律）

给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。

看了题解做出来的，首先需要转一圈涉及的四个位置：matrix\[i][j], matrix\[len - 1 - j][i], matrix\[len - i - 1][len - j - 1], matrix\[j][len - 1 - i]，在这四个位置之间进行值的交换，（这个还是比较好想到的）

还有一个比较重要的点是，因为每次修改4个位置，所以把矩阵分割成4块，只需要枚举一块上[i, j]的就行，如果n为奇数，就把最中心的那个空出来

具体的看题解吧

我的代码如下：

```java
class Solution {
    public void rotate(int[][] matrix) {
        int len = matrix.length;
        for(int i = 0; i < len / 2; i++){ // 什么都不做就是向下取整
            for(int j = 0; j < (len + 1) / 2; j++){ // 加上1，向上取整
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[len - 1 - j][i];
                matrix[len - 1 - j][i] = matrix[len - i - 1][len - j - 1];
                matrix[len - i - 1][len - j - 1] = matrix[j][len - 1 - i];
                matrix[j][len - 1 - i] = tmp;
            }
        }
    }
}
```



#### 49 字母异位词分组

给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

示例 1:

```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

示例 2:

```
输入: strs = [""]
输出: [[""]]
```

这道题不难，就是一眼看过去不太确定要用什么数据结构来做。

这里选的是key为String，value为List的HashMap

key存的是异位词排序后的结果

比较巧妙的是返回结果的时候用的`return new ArrayList<>(map.values());`

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for(int i = 0; i < strs.length; i++){
            char[] chars = strs[i].toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(strs[i]);
            map.put(key, list);
        }
        return new ArrayList<>(map.values());
    }
}
```



#### 50 Pow(x,n)（快速幂，利用二进制）

实现 [pow(*x*, *n*)](https://www.cplusplus.com/reference/valarray/pow/) ，即计算 `x` 的 `n` 次幂函数（即，`x^n` ）。

大致思路就是把n转为2进制，比如n=10, 二进制为1010，在循环时x不断地平方，x=x, x^2, x^4, x^8，当二进制位上是1时，就乘上x, n=10对应的就是x^8 * x^2

对于n为负数的，先转为正数，最后取倒数，需要注意-2147483648取反后会越界

https://leetcode-cn.com/problems/divide-two-integers/solution/tong-ge-lai-shua-ti-la-bei-zeng-cheng-fa-6qbg/ 这个讲得不错

https://leetcode-cn.com/problems/divide-two-integers/solution/gong-shui-san-xie-dui-xian-zhi-tiao-jian-utb9/

```java
class Solution {
    public double myPow(double x, int n) {
        boolean flag = true;
        if(n == 0) return 1;
        long n_ = n; // 处理 n = -2147483648 的情况
        if(n < 0){
            flag = false;
            n_ = -n_;
        }
        double res = 1;
        while(n_ != 0){
            if(n_ % 2 == 1){
                res *= x;
            }
            x *= x; 
            n_ /= 2;
        }
        if(!flag){
            res = 1 / res;
        }
        return res;
    }
}
```



#### 51 N皇后

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

```
输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
解释：如上图所示，4 皇后问题存在两个不同的解法。
```

这道题的回溯部分其实不难，对我来说比较难的是生成这个棋盘对应的List\<List\<String>>

看了题解的做法，是用一个int[]先记录每一行选的是哪一列，然后再用一个函数把这个数组转为List\<List\<String>>

题解还用了set来判断是否能放，我用的是boolean数组，比用set快了不少

+ 对角线的规律需要稍微找一下（一个是行加列，一个是行减列）

```java
class Solution {
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        int[] queens = new int[n];
        boolean[] columns = new boolean[n];
        boolean[] diagonals1 = new boolean[2 * n - 1];
        boolean[] diagonals2 = new boolean[2 * n - 1];
        dfs(res, queens, n, 0, columns, diagonals1, diagonals2);
        return res;
    }

    public void dfs(List<List<String>> res, int[] queens, int n, int row, boolean[] columns, boolean[] diagonals1, boolean[] diagonals2) {
        if (row == n) {
            List<String> board = generateBoard(queens, n);
            res.add(board);
            return;
        } 
        for (int i = 0; i < n; i++) {
            if(columns[i]) continue;
            int diagonal1 = row - i + n - 1;
            if (diagonals1[diagonal1]) continue;
            int diagonal2 = row + i;
            if (diagonals2[diagonal2]) continue;
            queens[row] = i;
            columns[i] = true;
            diagonals1[diagonal1] = true;
            diagonals2[diagonal2] = true;
            dfs(res, queens, n, row + 1, columns, diagonals1, diagonals2);
            // queens[row] = 0; 这个可写可不写
            columns[i] = false;
            diagonals1[diagonal1] = false;
            diagonals2[diagonal2] = false;
        }
        return;
    }

    public List<String> generateBoard(int[] queens, int n) {
        List<String> board = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            char[] row = new char[n];
            Arrays.fill(row, '.'); // 注意语法！
            row[queens[i]] = 'Q';
            board.add(new String(row));// 注意语法！
        }
        return board;
    }
}
```



#### 52 N皇后二

给你一个整数 `n` ，返回 **n 皇后问题** 不同的解决方案的数量。

```java
class Solution {
    int res = 0; // 注意！res如果作为参数的话是传值不是传引用，所以把它作为类变量了
    public int totalNQueens(int n) {
        boolean[] columns = new boolean[n];
        boolean[] diagonals1 = new boolean[2 * n - 1];
        boolean[] diagonals2 = new boolean[2 * n - 1];
        dfs(n, 0, columns, diagonals1, diagonals2);
        return res;
    }

    public void dfs(int n, int row, boolean[] columns, boolean[] diagonals1, boolean[] diagonals2) {
        if (row == n) {
            res++;
            return;
        } 
        for (int i = 0; i < n; i++) {
            if(columns[i]) continue;
            int diagonal1 = row - i + n - 1;
            if (diagonals1[diagonal1]) continue;
            int diagonal2 = row + i;
            if (diagonals2[diagonal2]) continue;
            
            columns[i] = true;
            diagonals1[diagonal1] = true;
            diagonals2[diagonal2] = true;

            dfs(n, row + 1, columns, diagonals1, diagonals2);
            
            columns[i] = false;
            diagonals1[diagonal1] = false;
            diagonals2[diagonal2] = false;
        }
        return;
    }
}
```



#### 53 最大连续子序列和（DP）

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。和剑指offer42相同

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

其实可以不用dp数组，因为总是只关注前一个，所以用一个int变量存就行了



#### 54 螺旋矩阵

给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int[][] directions = {{0,1},{1,0},{0,-1},{-1,0}};
        int m = matrix.length;
        int n = matrix[0].length;
        List<Integer> res = new ArrayList<>();
        int x = 0, y = 0;
        int direction = 0;
        boolean[][] visited = new boolean[m][n];
        for(int i = 0; i < m * n; i++){
            res.add(matrix[x][y]);
            visited[x][y] = true;
            int xTemp = x + directions[direction][0];
            int yTemp = y + directions[direction][1];
            if(xTemp < 0 || yTemp < 0 || xTemp >= m || yTemp >= n || visited[xTemp][yTemp]){
                direction = (direction + 1) % 4;
                x += directions[direction][0];
                y += directions[direction][1];
            }
            else{
                x = xTemp;
                y = yTemp;
            }
        }
        return res;
    }
}
```



#### 55 跳跃游戏（和45题类似）

给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

示例 1：

```
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
```


提示：

+ 1 <= nums.length <= 3 * 104
+ 0 <= nums[i] <= 105

```java
class Solution {
    public boolean canJump(int[] nums) {
        int end = 0;
        for(int i = 0; i <= end; i++){
            end = Math.max(end, nums[i] + i);
            if(end >= nums.length - 1) return true;
        }
        return false;
    }
}
```

和45题的区别是不需要用step变量来记忆走了几步，还是需要end来存当前是不是可以出发的位置，这个很重要，我们必须在可以走的范围内前进，同时更新可以走的范围，直到这个范围涵盖了终点



#### 56 合并区间（先排序）

以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间。 

示例 1：

```
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

示例 2：

```
输入：intervals = [[1,4],[4,5]]
输出：[[1,5]]
解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
```


提示：

+ 1 <= intervals.length <= 104
+ intervals[i].length == 2
+ 0 <= starti <= endi <= 104

最开始感觉有点难，其实只要排序再遍历一次就可以了。

排序是按区间的起点排的，遍历的时候就看下能不能和前面的合并，可以的话就把前一个区间扩张，思路挺简单的，不要想复杂了

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>(){
            public int compare(int[] o1, int[] o2){
                return o1[0] - o2[0];
            }
        });
        LinkedList<int[]> res = new LinkedList<>();
        res.add(intervals[0]);
        for(int i = 1; i < intervals.length; i++){
            if(intervals[i][0] <= res.peekLast()[1]){
                if(intervals[i][1] > res.peekLast()[1]){
                    res.peekLast()[1] = intervals[i][1];
                }
            }
            else{
                res.add(intervals[i]);
            }
        }
        return res.toArray(new int[res.size()][]);
    }
}
```

+ 语法：`res.toArray(new int[res.size()][]);`



#### 57 插入区间（做了特别久的一道题）

给你一个无重叠的，按照区间起始端点排序的区间列表。

在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

示例 1：

```
输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
输出：[[1,5],[6,9]]
```

示例 2：

```
输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出：[[1,2],[3,10],[12,16]]
解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
```

示例 3：

```
输入：intervals = [], newInterval = [5,7]
输出：[[5,7]]
```

示例 4：

```
输入：intervals = [[1,5]], newInterval = [2,3]
输出：[[1,5]]
```

示例 5：

```
输入：intervals = [[1,5]], newInterval = [2,7]
输出：[[1,7]]
```


提示：

+ 0 <= intervals.length <= 104
+ intervals[i].length == 2
+ 0 <= intervals\[i][0] <= intervals\[i][1] <= 105
+ intervals 根据 intervals\[i][0] 按 升序 排列
+ newInterval.length == 2
+ 0 <= newInterval[0] <= newInterval[1] <= 105

一开始想用二分查找来做，写了一半感觉还是有点烦，边界问题太多了，遂放弃看了一眼题解。

然后自己又极其扭曲地写出了一版。本来是一个for循环处理每个的，写了一半我把它拆成了三个while循环，后面再根据几种特殊情况又改来改去，只能说很混乱

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {

        List<int[]> res = new ArrayList<>();
        int i = 0;
        boolean flag = false;
        // 第一阶段
        while(i < intervals.length && intervals[i][1] < newInterval[0]){
            res.add(intervals[i]);
            i++;
        }
        // 第二阶段    
        int j = i;
        while(j < intervals.length && intervals[j][0] <= newInterval[1]) j++;
		// while循环结束时j应该是第三阶段的第一个，i是第二阶段的第一个
        if(i < intervals.length){
            if(j - 1 >= 0){
                int[] merge = {Math.min(newInterval[0], intervals[i][0]), Math.max(newInterval[1], intervals[j - 1][1])};
                res.add(merge);
            }
            // 如果j等于0，说明第一第二阶段都不存在，这时候要先加newInterval
            else res.add(newInterval);
            flag = true;
        }    
		// 第三阶段
        while(j < intervals.length){
            res.add(intervals[j]);
            j++;
        }    
        
        if(!flag) res.add(newInterval);
        return res.toArray(new int[res.size()][]);
    }

}
```

需要考虑的边界

+ intervals为空
+ 没有第一阶段
+ 没有第二阶段
+ 没有第三阶段

就把思路搞得很乱，没有把几种情况统一起来。后来又去看了题解，感觉题解的做法思路清晰非常多

https://leetcode-cn.com/problems/insert-interval/solution/cha-ru-qu-jian-by-leetcode-solution/

就是一个for循环，是第一阶段还是第三阶段都很好判断，剩下的就肯定是第二阶段了，如果是第二阶段，就更新newInterval的值就可以了

```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> res = new ArrayList<>();
        boolean added = false;
        for(int i = 0; i < intervals.length; i++){
            // 第一阶段
            if(intervals[i][1] < newInterval[0]) res.add(intervals[i]);
            // 第三阶段
            else if(intervals[i][0] > newInterval[1]){
                // 如果newInterval还没加入，需要在第三阶段加入前加入
                if(!added){
                    res.add(newInterval);
                    added = true;
                }
                res.add(intervals[i]);
            } 
            // 第二阶段
            else{
                newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
                newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            }
        }
        // 如果newInterval还没加入
        if(!added) res.add(newInterval);
        return res.toArray(new int[res.size()][]);
    }

}
```



#### 58 最后一个单词的长度

给你一个字符串 s，由若干单词组成，单词之间用空格隔开。返回字符串中最后一个单词的长度。如果不存在最后一个单词，请返回 0 。

单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。

```java
class Solution {
    public int lengthOfLastWord(String s) {
        // 第一种，效率低
        // String[] strings = s.split(" ");
        // for(int i = strings.length - 1; i >= 0; i--){
        //     if(strings[i] == "") continue;
        //     return strings[i].length();
        // }
        // return 0;
        
        // 第二种，效率高
        s = s.trim();
        for(int i = s.length() - 1; i >= 0; i--){
            if(s.charAt(i) == ' ') return s.length() - i - 1;
        }
        return s.length();
    }
}
```

需要注意的是第二种做法在处理"  " 和 "a"这两种特例时都是在最后返回删除空格后的s的长度，很神奇



#### 59 螺旋矩阵二（类似54题）

给你一个正整数 `n` ，生成一个包含 `1` 到 `n2` 所有元素，且元素按顺时针顺序螺旋排列的 `n x n` 正方形矩阵 `matrix` 。

这道题比54题还省了一个visited数组

```java
class Solution {
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int i = 0, j = 0, num = 1;
        int[][] directions = {{0,1}, {1, 0}, {0, -1}, {-1, 0}};
        int direction = 0;
        while(num <= n * n){
            res[i][j] = num;
            num++;
            int i_ = i + directions[direction][0];
            int j_ = j + directions[direction][1];
            // 第一次把if里的i_写成了i，导致了数组越界，建议还是不要这样命名
            if(i_ >= n || j_ >= n || i_ < 0 || j_ < 0 || res[i_][j_] != 0){
                direction = (direction + 1) % 4;
                i += directions[direction][0];
                j += directions[direction][1]; 
            }
            else{
                i = i_;
                j = j_;
            }
        }
        return res;
    }
}
```



#### 60 排序序列（hard, 可以用回溯）

给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。

按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：

"123"
"132"
"213"
"231"
"312"
"321"
给定 n 和 k，返回第 k 个排列。

示例 1：

```
输入：n = 3, k = 3
输出："213"
```

示例 2：

```
输入：n = 4, k = 9
输出："2314"
```

示例 3：

```
输入：n = 3, k = 1
输出："123"
```


提示：

+ 1 <= n <= 9
+ 1 <= k <= n!

一开始没啥思路，后来看了题解 https://leetcode-cn.com/problems/permutation-sequence/solution/hui-su-jian-zhi-python-dai-ma-java-dai-ma-by-liwei/，发现可以用类似回溯/深搜的做法来做，只不过不需要“回”。而是尽可能地剪枝，在进入这个分支之前先判断下第k个有没有在这个分支里，如果没有就直接跳过这个分支。如果有就递归调用函数。调用过函数就直接返回了，因为不需要考虑其他树枝了。

因为涉及到阶乘，所以可以存一个阶乘数组，就不用重复计算了

```java
class Solution {
    int[] factorial;
    boolean[] visited;
    int n;
    int k;
    public String getPermutation(int n, int k) {
        this.n = n;
        this.k = k;
        factorial = new int[n];
        // 记得调用这个函数
        setFactorial(n);
        visited = new boolean[n + 1];
        StringBuilder res = new StringBuilder();
        dfs(res, 0);
        return res.toString();
    }
    void dfs(StringBuilder res, int count){
        if(res.length() == n) return;

        int curFactorial = factorial[n - 1 - count];
        for(int i = 1; i <= n; i++){
            if(visited[i]) continue;
            if(curFactorial < k){
                k -= curFactorial;
                continue;
            }
            else{
                res.append(i);
                visited[i] = true;
                dfs(res, count + 1);
                // 这里记得加return
                return;
            }
        }
    }
    void setFactorial(int n){
        factorial[0] = 1;
        for(int i = 1; i < n; i++){
            factorial[i] = factorial[i - 1] * i;
        }
    }
}
```

这么做很形象，而且可以利用回溯的模板，速度也挺快的，不过没错k只能减的，用除的会更快

空间复杂度O(N)，时间复杂度O(N2)

还可以用除法，可以更快一点点，但是除法的写法有更多细节的地方可能出错

```java
import java.util.LinkedList;
import java.util.List;

public class Solution {

    public String getPermutation(int n, int k) {
        // 注意：相当于在 n 个数字的全排列中找到下标为 k - 1 的那个数，因此 k 先减 1
        k --;

        int[] factorial = new int[n];
        factorial[0] = 1;
        // 先算出所有的阶乘值
        for (int i = 1; i < n; i++) {
            factorial[i] = factorial[i - 1] * i;
        }

        // 这里使用数组或者链表都行
        List<Integer> nums = new LinkedList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }

        StringBuilder stringBuilder = new StringBuilder();

        // i 表示剩余的数字个数，初始化为 n - 1
        for (int i = n - 1; i >= 0; i--) {
            int index = k / factorial[i] ;
            stringBuilder.append(nums.remove(index));
            k -= index * factorial[i];
        }
        return stringBuilder.toString();
    }
}
```



#### 61 旋转链表

给你一个链表的头节点 `head` ，旋转链表，将链表每个节点向右移动 `k` 个位置。

我的思路是先遍历一遍求出链表里有几个结点，并且将表尾接到表头变成一个环，然后k需要先对结点数求模。比如有3个结点的链表，k = 4就等同于k = 1，k = 3就等同于k = 0。然后还需要用count -  k % count，这个是我找规律得出的，可以理解为，新的头是在旧的头往左走k % count步，往左走x步就是往右走    count - x 步。就可以得到新的头啦，在新的尾处断开就行。（所以循环结束拿到的是新的尾，然后返回它的next就是新的头）因为k = 0时 count - k == count，所以再加一个取模。 

```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null) return null;
        int count = 1;
        ListNode iter = head;
        while(iter.next != null){
            iter = iter.next;
            count++;
        }
        // 循环结束时iter为链表的尾结点
        iter.next = head;
        int mod = (count - k % count) % count;
        
        while(mod != 0){
            iter = iter.next;
            mod--;
        }

        ListNode newHead = iter.next;
        iter.next = null;
        return newHead;
    }
}
```



#### 62 不同路径（二维DP/组合数学）

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

最先想到的是组合的做法，机器人一共走m+n-2步，其中m-1步是向右，这就是一个m+n-2选m-1的问题

但是我卡在了求组合结果，因为乘积会超出范围，所以我用了bigInteger，后来看题解发现它用的long就能做了，我也改成long还是写错，最后不得不做得和题解一模一样。我最迷惑的是一边乘一边除中间的时候为啥能整除。

能整除是因为它分子和分母都是从小到大累积的，每一步中间结果都是一个组合数，所以能整除

```java
import java.math.BigInteger; 
class Solution {
    // public int uniquePaths(int m, int n) {
    //     int i = m + n - 2;
    //     BigInteger res = BigInteger.valueOf(1);
    //     for(int j = m - 1; j >= 1; j--){
    //         res = res.multiply(BigInteger.valueOf(i));
    //         i--;
    //     }
    //     for(int j = m - 1; j >= 1; j--){
    //         res = res.divide(BigInteger.valueOf(j));
    //     }
    //     return res.intValue();
    // }
    public int uniquePaths(int m, int n){
        long ans = 1;
        for (int x = n, y = 1; y < m; ++x, ++y) {
            ans = ans * x / y;
        }
        return (int) ans;
    }
}
```

二维DP又是一开始完全没往那边想。其实二维DP非常好做的，dp\[i][j]表示的是从起点走到(i,j)的路径数，dp\[i][j]=dp\[i-1][j]+dp\[i][j-1]

为了节省空间，其实可以用一维数组来存，dp\[i-1][j]就是dp\[i][j]原来的值，所以可以省略i，写法也不难，如下：

```java
class Solution {
    public int uniquePaths(int m, int n){
        int[] dp = new int[n];
        dp[0] = 1;
        for(int i = 0; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[j] += dp[j - 1];
            }
        }
        return dp[n - 1];
    }
}
```



#### 63 不同路径二（二维DP，部分格子不能走）

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        if(obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1) return 0;
        int[][] dp = new int[m][n];
        dp[0][0] = 1;
        for(int j = 1; j < n; j++){
            dp[0][j] = obstacleGrid[0][j] == 1 ? 0 : dp[0][j - 1];
        }
        for(int i = 1; i < m; i++){
            dp[i][0] = obstacleGrid[i][0] == 1 ? 0 : dp[i - 1][0];
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                if(obstacleGrid[i][j] == 1){
                    dp[i][j] = 0;
                }
                else{
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                } 
            }
        }
        return dp[m - 1][n - 1];
    }
}
```



#### 64 最小路径和（二维DP）

给定一个包含非负整数的 `m x n` 网格 `grid` ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

**说明：**每次只能向下或者向右移动一步。

看完题目愣是没想到二维DP，脑子里一直想贪心或者递归···一看题解恍然大悟，明明是很简单的题

dp\[i][j]表示的是从起点走到(i,j)的最小路径和，做法就很类似求字符串的编辑距离，而且更简单。最后的结果就是dp\[m-1][n-1]，一开始一直在想怎么直接求出这两个点之间的距离，根本没想到要把从起点到每个点的距离求出来

```java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for(int i = 1; i < m; i++){
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for(int j = 1; j < n; j++){
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }
}
```



#### 65 有效数字（模拟题，没有看起来难）

有效数字（按顺序）可以分成以下几个部分：

一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分有效数字列举如下：

["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"]
部分无效数字列举如下：

["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"]
给你一个字符串 s ，如果 s 是一个 有效数字 ，请返回 true 。



大致的思路其实挺简单的，就是先扫描一遍看下有没有e，如果有，就分割成两半判断，没有就整个判断。另外，e不能超过1个。接下来就是判断是不是整数或者小数。整数的话就是除了最前面的正负号以外只能是数字，小数就是除了最前面的正负号以外，剩下的只能是数字和一个小数点。所以整数和小数的区别只是能不能出现一个小数点

```java
class Solution {
    public boolean isNumber(String s) {
        int len = s.length();
        char[] chars = s.toCharArray();
        int splitIndex = -1;
        for(int i = 0; i < len; i++){
            if(chars[i] == 'e' || chars[i] == 'E'){
                if(splitIndex == -1){
                    splitIndex = i;
                }
                // 如果前面已经出现过e了，返回false
                else return false;
            } 
        }
        if(splitIndex == -1){
            return check(chars, 0, len - 1, false);
        }
        else{
            if(!check(chars, 0, splitIndex - 1, false)) return false;
            // e的右边必须是整数
            return check(chars, splitIndex + 1, len - 1, true);
        }
    }
    boolean check(char[] chars, int start, int end, boolean mustInteger){
        if(start > end) return false;
        boolean hasDot = false;
        boolean hasDigit = false;
        if(chars[start] == '+' || chars[start] == '-') start++;
        for(int i = start; i <= end; i++){
            if(chars[i] >= '0' && chars[i] <= '9') hasDigit = true;
            else if(chars[i] == '.'){
                // 如果前面已经有一个小数点了，或者这个必须是整数，就返回false
                if(hasDot || mustInteger) return false;
                else hasDot = true;
            }
            else return false;
        }
        // 不管是整数还是小数都必须有数字
        return hasDigit;
    }
}
```



#### 66 加一（模拟）

给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。

最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。

你可以假设除了整数 0 之外，这个整数不会以零开头。

大致思路是，从后往前遍历每一位，如果遇到不等于9的就加一，跳出循环，返回结果。如果遇到9，就设为0，继续遍历。

如果循环结束时第一个数字是0，说明原来的数字是9...9这样的，最终结果应该是10...0这样的

```java
class Solution {
    public int[] plusOne(int[] digits) {
        for(int i = digits.length - 1; i >= 0; i--){
            if(digits[i] != 9){
                digits[i]++;
                break;
            }
            else{
                digits[i] = 0;
            }
        }
        
        if(digits[0] == 0){
            int[] res = new int[digits.length + 1];
            res[0] = 1;
            return res;
        }
        else return digits;
    }
}
```



#### 67 二进制求和（43题）

给你两个二进制字符串，返回它们的和（用二进制表示）。

输入为 **非空** 字符串且只包含数字 `1` 和 `0`。

借鉴了43题的做法

```java
class Solution {
    public String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int add = 0;
        while(i >= 0 || j >= 0 || add != 0){
            int digit1 = (i >= 0) ? (a.charAt(i) - '0') : 0;
            int digit2 = (j >= 0) ? (b.charAt(j) - '0') : 0;
            int sum = digit1 + digit2 + add;
            add = sum / 2;
            res.append(sum % 2);
            i--;
            j--;
        }
        return res.reverse().toString();
    }
}
```



#### 68 文本左右对齐（模拟）

给定一个单词数组和一个长度 maxWidth，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。

你应该使用“贪心算法”来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。

要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。

文本的最后一行应为左对齐，且单词之间不插入额外的空格。

说明:

+ 单词是指由非空格字符组成的字符序列。
+ 每个单词的长度大于 0，小于等于 maxWidth。
+ 输入单词数组 words 至少包含一个单词。

示例 1:

```
输入:
words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
输出:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
```

示例 2:

```
输入:
words = ["What","must","be","acknowledgment","shall","be"]
maxWidth = 16
输出:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
解释: 注意最后一行的格式应为 "shall be    " 而不是 "shall     be",
     因为最后一行应为左对齐，而不是左右两端对齐。       
     第二行同样为左对齐，这是因为这行只包含一个单词。
```

执行用时0ms，但是内存消耗比较多，我的代码如下：

```java
class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> res = new ArrayList<>();
        
        for(int i = 0; i < words.length; ){
            StringBuilder strb = new StringBuilder();
            strb.append(words[i]);
            int count = words[i].length();
            int j = i + 1;
            while(j < words.length && count + 1 + words[j].length() <= maxWidth){
                count = count + 1 + words[j].length();
                j++;
            }
            // 这个变量记录了如果每两个单词间只有一个空格，这行还剩多少空格没放
            int blank = maxWidth - count;

            // 如果这一行只能放下一个单词或者这是最后一行
            if(j - i == 1 || j == words.length){
                // 如果是最后一行，采用左对齐的方式，前面单词间隔一个空格
                if(j == words.length){
                    for(int k = 0; k < j - i - 1; k++){
                        strb.append(' ');
                        strb.append(words[i + k + 1]);
                    }
                }
                // 最后一行和这行只有一个单词的情况，都是在末尾把剩下的空格填上
                char[] curBlank = new char[blank];
                Arrays.fill(curBlank, ' ');
                strb.append(curBlank);
            }
            
            else{
                // 这个数组记录的是单词之间要放几个空格
                int[] blanks = new int[j - i - 1];
                // 初始化为1个空格加上平均分的空格数，保证大家的空格数目平均
                Arrays.fill(blanks, 1 + blank / (j - i - 1));
                // 余下来的空格从左往右分，保证左边的空格比右边的多
                int remain = blank % (j - i - 1);
                for(int k = 0; k < remain; k++){
                    blanks[k]++;
                }
                // 把空格和单词填上
                for(int k = 0; k < blanks.length; k++){
                    char[] curBlank = new char[blanks[k]];
                    Arrays.fill(curBlank, ' ');
                    strb.append(curBlank);
                    strb.append(words[i + k + 1]);
                }
            }
            
            res.add(strb.toString());
            i = j;
        }

        return res;
    }
}
```



#### 69 sqrt(x) （有点难的二分查找）

给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。

一开始傻傻地没有想到二分，就递减地找，看了题解才反应过来

第二次看发现好多边界问题啊



```java
class Solution {
    public int mySqrt(int x) {
        int low = 0, high = x, res = 0;
        while(low <= high){
            int mid = low + (high - low) / 2;
            if((long)mid * mid <= x){
                low = mid + 1;
                res = mid;
            } 
            else{
                high = mid - 1;
            }
        }
        return res;
    }
}
```



#### 70 爬楼梯

假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

注意：给定 *n* 是一个正整数。

最经典的递归了，一开始提交报错了，因为没有考虑到n=1的情况，对dp[2]赋值会数组越界

```java
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= n; i++){
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```

优化过空间的如下

```java
class Solution {
    public int climbStairs(int n) {
        if(n == 1) return 1;
        int pre1 = 1;
        int pre2 = 1;
        int res = 0;
        for(int i = 2; i <= n; i++){
            res = pre1 + pre2;
            pre2 = pre1;
            pre1 = res;
        }
        return res;
    }
}
```



#### 71 简化路径（涉及到比较多语法的模拟题）

给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为更加简洁的规范路径。

在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。任意多个连续的斜杠（即，'//'）都被视为单个斜杠 '/' 。 对于此问题，任何其他格式的点（例如，'...'）均被视为文件/目录名称。

请注意，返回的 规范路径 必须遵循下述格式：

始终以斜杠 '/' 开头。
两个目录名之间必须只有一个斜杠 '/' 。
最后一个目录名（如果存在）不能 以 '/' 结尾。
此外，路径仅包含从根目录到目标文件或目录的路径上的目录（即，不含 '.' 或 '..'）。
返回简化后得到的 规范路径 。

这道题没有想象中难，其实思路挺简单的

先按照 / 将字符串分割，再对分割后得到的数组进行遍历，如果当前字符串为空或者一个点，continue，如果为两个点，说明是上一级，就pop，其余情况就push。

因为最后还要按顺序把结果拼接起来，所以需要一个双端队列

```java
class Solution {
    public String simplifyPath(String path) {
        String[] list = path.split("/"); // 这里返回的是数组不是列表
        Deque<String> queue = new LinkedList<>(); // 这里要用双端队列
        for(int i = 0; i < list.length; i++){
            String cur = list[i];
            if(cur.length() == 0 || cur.equals(".")) continue;
            else if(cur.equals("..")){
                if(!queue.isEmpty()) queue.removeLast();
            } 
            else queue.addLast(cur);
        }
        StringBuilder strb = new StringBuilder();
        while(!queue.isEmpty()){
            strb.append("/");
            strb.append(queue.removeFirst());
        }
        // 这里要考虑结果就是根目录的情况
        if(strb.length() == 0) strb.append("/");
        return strb.toString();
    }
}
```



#### 72 编辑距离

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符

```java
class Solution {
    public int minDistance(String word1, String word2) {
        word1 = "_" + word1;
        word2 = "_" + word2;
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dp = new int[len1][len2];
        for(int i = 0; i < len1; i++){
            dp[i][0] = i;
        }
        for(int j = 1; j < len2; j++){
            dp[0][j] = j;
        }
        for(int i = 1; i < len1; i++){
            for(int j = 1; j < len2; j++){
                if(word1.charAt(i) == word2.charAt(j)){
                    dp[i][j] = dp[i - 1][j - 1];
                }
                else{
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                }
            }
        }
        return dp[len1 - 1][len2 - 1];
    }
}
```



#### 73 矩阵置零（扫描两遍）

给定一个 `m x n` 的矩阵，如果一个元素为 **0** ，则将其所在行和列的所有元素都设为 **0** 。请使用 **[原地](http://baike.baidu.com/item/原地算法)** 算法

题目要求用常量空间来解决

这道题一开始没意识到难点，想说就一边遍历一边改，后来意识到不行，这样没法区分哪些0是原来的，哪些是被改的，而且时间复杂度也不低

看了一下题解 https://leetcode-cn.com/problems/set-matrix-zeroes/solution/ju-zhen-zhi-ling-by-leetcode-solution-9ll7/，反应过来是先扫描一遍，把哪一列哪一行有0记录下来，第二次扫描的时候再把他们置0。这样的话空间复杂度是O(M+N)

我觉得这样其实就可以了，用两个数组记录下行和列是否需要置0，思路非常清晰

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean[] row = new boolean[m];
        boolean[] col = new boolean[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = col[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
}
```

如果要用常数的空间的话，就要利用第一行和第一列来存剩下每一列和每一行是否有0，这样就需要两个变量来记录第一行和第一列本来是否有0，思路有点绕，需要想一下

```java
class Solution {
    public void setZeroes(int[][] matrix) {
        boolean rowMark = false;
        boolean columnMark = false;
        int m = matrix.length;
        int n = matrix[0].length;
        for(int i = 0; i < m; i++){
            if(matrix[i][0] == 0){
                columnMark = true;
                break;
            } 
        } 
        for(int j = 0; j < n; j++){
            if(matrix[0][j] == 0){
                rowMark = true;
                break;
            }
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                if(matrix[i][j] == 0){
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        for(int i = 1; i < m; i++){
            for(int j = 1; j < n; j++){
                if(matrix[0][j] == 0 || matrix[i][0] == 0){
                    matrix[i][j] = 0;
                }
            }
        }
        if(rowMark){
            for(int j = 0; j < n; j++){
                matrix[0][j] = 0;
            }
        }
        if(columnMark){
            for(int i = 0; i < m; i++){
                matrix[i][0] = 0;
            }
        }
    }
}
```



#### 74 搜索二维矩阵（二分查找比target小的最大的数）

编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。

一开始只想到了从左下角开始找的O(M+N)的做法，看了题解才想到可以用两次二分查找

第一次二分找到target所在的行，这一行的行首应该是第一列中小于等于target的最大数

第二次二分就是在这一行里找

```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = searchRow(target, matrix);
        if(row == -1) return false;
        return searchColumn(target, matrix, row);
    }

    int searchRow(int target, int[][] matrix){
        int low = -1; // 注意这里要初始化为-1，表示-1行的行首是负无穷
        int high = matrix.length - 1;
        while(low < high){ // low == high的时候跳出循环是因为这时的肯定是答案（可以证明）
            int mid = low + (high - low) / 2 + 1; // 这里的加一是为了避免死循环
            if(matrix[mid][0] == target) return mid;
            else if(matrix[mid][0] < target) low = mid;
            else high = mid - 1;
        }
        return low;
    }

    boolean searchColumn(int target, int[][] matrix, int row){
        int low = 0; 
        int high = matrix[0].length - 1;
        while(low <= high){
            int mid = low + (high - low) / 2;
            if(matrix[row][mid] == target) return true;
            else if(matrix[row][mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return false;
    }
}
```

+ 犯的一个小错误：high初始化成length了，忘记减一了



#### 75 颜色分类（三指针，partition，易错）

给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库的sort函数的情况下解决这个问题。



我把我的做法称为三指针，不过题解里称为双指针

提交错了两次，改了好几遍。一开始用low来遍历，mid和high标记的是接下来mid和high要放的位置

后来改成了用mid遍历，low和high标记的是目前最右边的0和最左边的2的位置，其实思路都差不多，就是一些边界情况很容易出错

```java
class Solution {
    public void sortColors(int[] nums) {
        int low = -1, mid = 0, high = nums.length;
        while(mid < high){
            if(nums[mid] == 0){
                low++;
                nums[mid] = nums[low];
                nums[low] = 0;
                mid++; // 注意这里要加一，nums[mid]为1或0，所以可以直接加一
            }
            else if(nums[mid] == 2){
                high--;
                nums[mid] = nums[high];
                nums[high] = 2;    
            }
            else{
                mid++;
            }
        }
    }
}
```



#### 76 最小覆盖子串（滑动窗口，气死我了）

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。 

注意：

+ 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
+ 如果 s 中存在这样的子串，我们保证它是唯一的答案。


示例 1：

```
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
```

示例 2：

```
输入：s = "a", t = "a"
输出："a"
```

示例 3:

```
输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。
```


提示：

1 <= s.length, t.length <= 105
s 和 t 由英文字母组成


进阶：你能设计一个在 o(n) 时间内解决此问题的算法吗？

滑动窗口，两重while循环，稍微有点难想的是怎么判断窗口中的字符满足要求了，这里是当两个map的一个key所对应的value相等的时候就给count加一，当count等于map t 的size时就说明所有key都满足要求了

**最后重点** 栽在了Integer没有用equals来判等，找这个bug找了一小时，气死

```java
class Solution {
    public String minWindow(String s, String t) {
        Map<Character, Integer> mapt = new HashMap<>();
        Map<Character, Integer> maps = new HashMap<>();
        for(int i = 0; i < t.length(); i++){
            mapt.put(t.charAt(i), mapt.getOrDefault(t.charAt(i), 0) + 1);
        }
        int left = 0;
        int right = 0;
        int count = 0;
        int start = -1;
        int len = Integer.MAX_VALUE;
        while(right < s.length()){
            char cur = s.charAt(right);
            if(mapt.containsKey(cur)){
                maps.put(cur, maps.getOrDefault(cur, 0) + 1);  
                // 这里一定要用equals，不然会错
                if(maps.get(cur).equals(mapt.get(cur))){
                    count++;
                } 
            }
            while(count == mapt.size()){
                if(right - left + 1 < len) {
                    start = left;
                    len = right - left + 1;
                }
                char del = s.charAt(left);
                if(mapt.containsKey(del)){
                    if(maps.get(del).equals(mapt.get(del))) count--;
                    maps.put(del, maps.get(del) - 1);
                }
                left++;
            }
            right++;
        }
        if(start == -1) return "";
        else return s.substring(start, start + len);
    }
}
```

**改进** 别用map了，用int数组会更好，要注意不是26，有大小写字母，58是我用z-A试出来的

因为是用的数组，里面会有些位置是零，所以不能用数组的长度和count比，换了一种比较方法和count的更新方法

count表示的含义变成了match的字母个数，例如s="a", t="aa"，count就是1，如果是前一种做法count是0

```java
class Solution {
    public String minWindow(String s, String t) {
        int[] countS = new int[58];
        int[] countT = new int[58];
        for(int i = 0; i < t.length(); i++){
            countT[t.charAt(i) - 'A']++;
        }
        int left = 0, right = 0;
        int start = -1, len = 1000000;
        int count = 0;
        while(right < s.length()){
            int cur = s.charAt(right) - 'A';
            if(countT[cur] > 0){
                countS[cur]++;
                if(countS[cur] <= countT[cur]) count++;
            }
            while(count == t.length()){
                if(right - left + 1 < len){
                    start = left;
                    len = right - left + 1;
                }
                int del = s.charAt(left) - 'A';
                if(countT[del] > 0){
                    countS[del]--;
                    if(countS[del] < countT[del]) count--;
                }
                left++;
            }
            right++;
        }
        return start == -1 ? "" : s.substring(start, start + len);
    }
}
```



#### 77 组合（需要剪枝的回溯）

给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。

你可以按 任何顺序 返回答案。

```
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

看了题解才知道可以通过剪枝缩短时间，举个例子，如果是n = 4, k = 3, 那一开始什么都没选的时候i最大是2，如果i是3的话后面根本就没有两个数可以选。但如果是已经选了一个数字，比如选了1，接下来还要选两个数字，那i就可以取到4了，同理，如果接下来只要再选一个，那i可以取到5。

所以i的范围和k，list的size，n有关。

```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        combine(res, list, k, n, 1);
        return res;
    }
    void combine(List<List<Integer>> res, List<Integer> list, int k, int n, int start){
        if(list.size() == k){
            res.add(new ArrayList<>(list));
            return;
        }
        // i的范围用到了剪枝
        for(int i = start; i <= n - k + list.size() + 1; i++){
            list.add(i);
            combine(res, list, k, n, i + 1);
            list.remove(list.size() - 1);
        }
    }
}
```



#### 78 子集（有点特别的回溯）

给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

示例 1：

```
输入：nums = [1,2,3]
输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

示例 2：

```
输入：nums = [0]
输出：[[],[0]]
```


提示：

+ 1 <= nums.length <= 10
+ -10 <= nums[i] <= 10
+ nums 中的所有元素 互不相同

刚开始看到只想到了二进制的做法和比较傻的回溯，看了题解才意识到回溯也可以不用判断list长度是否等于nums的长度，只要用一个index判断是不是到了就行。我太傻了

大致思路是，对每个位置都有选或者不选两种，选的话就加入list，进入递归，递归结束后从list里移除，正好对应了不选，然后进入递归，递归结束之后也不用再从list里移除，因为本来就没加进去。

```java
class Solution {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        dfs(res, list, 0, nums);
        return res;
    }
    void dfs(List<List<Integer>> res, List<Integer> list, int cur, int[] nums){
        if(cur == nums.length){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        list.add(nums[cur]);
        dfs(res, list, cur + 1, nums);
        list.remove(list.size() - 1);
        dfs(res, list, cur + 1, nums);
        // 这里和一般回溯不太一样，不需要再去掉一次了
    }
}
```

这道题不能用39题的做法，39题的做法会导致结果变成123，13，23，3



#### 79 单词搜索（回溯）

给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

```java
class Solution {
    char[][] board;
    boolean[][] visited;
    String word;
    int[][] directions = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    public boolean exist(char[][] board, String word) {
        // if(board.length == 0 || word.length() == 0)
        this.board = board;
        this.word = word;
        this.visited = new boolean[board.length][board[0].length];
        for(int i = 0; i < board.length; i++){
            for(int j = 0; j < board[0].length; j++){
                if(dfs(0, i, j)) return true;
            }
        }
        return false;
    }
    boolean dfs(int index, int i, int j){
        if(index == word.length()) return true;
        if(!isValid(i, j) || board[i][j] != word.charAt(index)) return false;
        visited[i][j] = true;
        for(int[] direction : directions){
            if(dfs(index + 1, i + direction[0], j + direction[1])) return true;
        }
        visited[i][j] = false;
        return false;
    } 
    boolean isValid(int i, int j){
        // 记得判断位置是否越界，是否已经访问过
        if(i < 0 || j < 0 || i >= board.length || j >= board[0].length || visited[i][j]) return false;
        else return true;
    }
}


```



#### 80 删除有序数组中的重复项二（26题的进阶）

给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使每个元素 最多出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。



一开始想了半天没想到怎么做，后来又看了半天26题的解法，才想到其实只要借助**两个变量，一个记录当前的数字，一个记录这个数字出现的次数**，就可以判断这个数字要不要留下，还有一个和26题相同的变量用来记录我应该把遍历到的数字放在哪

```java
class Solution {
    public int removeDuplicates(int[] nums) {
        int count = 1;
        int cur = nums[0];
        int nextPos = 1;
        for(int i = 1; i < nums.length; i++){
            if(nums[i] == cur && count == 1){
                nums[nextPos] = nums[i];
                nextPos++;
                count++;
            }
            else if(nums[i] != cur){
                nums[nextPos] = nums[i];
                nextPos++;
                cur = nums[i];
                count = 1;
            }
        }
        return nextPos;
    }
}
```



#### 81 搜索旋转排序数组二（33题的变体）

已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。

给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。

```
输入：nums = [2,5,6,0,0,1,2], target = 3
输出：false
```

刚看题目觉得很复杂，要讨论很多情况，看了https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/solution/zai-javazhong-ji-bai-liao-100de-yong-hu-by-reedfan/ 这个题解之后就非常清晰，其实只是在33题的基础上多加了一个if语句

> 10111 和 11101 这种。此种情况下 nums[start] == nums[mid]，分不清到底是前面有序还是后面有序，此时 start++ 即可。相当于去掉一个重复的干扰项。

然而在写代码的时候我还是出了好多小错误，说明我还是不够熟练

这是最终版本的代码

```java
class Solution {
    public boolean search(int[] nums, int target) {
        int low = 0;
        int high = nums.length - 1;
        while(low <= high){
            int mid = low + (high - low) / 2;
            if(target == nums[mid]) return true;
            if(nums[mid] == nums[low]) {
                low++;
            }
            else if(nums[mid] > nums[low]){
                if(target >= nums[low] && target < nums[mid]){
                    high = mid - 1;
                }
                else{
                    low = mid + 1;
                }
            }
            else{
                if(target > nums[mid] && target <= nums[high]){
                    low = mid + 1;
                }
                else{
                    high = mid - 1;
                }
            }
        }
        return false;
    }
}
```



#### 82 删除排序链表中的重复元素二

存在一个按升序排列的链表，给你这个链表的头节点 head ，请你删除链表中所有存在数字重复情况的节点，只保留原始链表中 没有重复出现 的数字。

返回同样按升序排列的结果链表。

为了更方便删除操作，我new了一个dummy head节点。注意循环终止条件

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null) return null;
        ListNode dummyHead = new ListNode(0, head);
        ListNode iter = dummyHead;
        while(iter.next != null && iter.next.next != null){
            if(iter.next.val == iter.next.next.val){
                int value = iter.next.val;
                ListNode iterNext = iter.next.next;
                while(iterNext != null && iterNext.val == value){
                    iterNext = iterNext.next;
                } 
                iter.next = iterNext;
            }
            else{
                iter = iter.next;
            }
        }
        return dummyHead.next;
    }
}
```



#### 83 删除排序链表中的重复元素

存在一个按升序排列的链表，给你这个链表的头节点 `head` ，请你删除所有重复的元素，使每个元素 **只出现一次** 。

返回同样按升序排列的结果链表。

```java
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        if(head == null) return null;
        ListNode iter = head;
        while(iter.next != null){
            if(iter.val == iter.next.val){
                iter.next = iter.next.next;
            }
            else{
                iter = iter.next;
            }     
        }
        return head;
    }
}
```



#### 84 柱状图中最大的矩形（单调递增栈）

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

根据[五分钟学算法](https://mp.weixin.qq.com/s?__biz=MzUyNjQxNjYyMg==&mid=2247498521&idx=2&sn=07e4126177f09aae483dbbbf3c64cde1&chksm=fa0d9498cd7a1d8e457d9e140eb7d31c4354f8380a070d6aa4df7839b05e21022e1e784d3db1&scene=126&sessionid=1616597425&key=4b3245558f41098e4b8a077ad5a361c77e7ab4dbd7f405cd9534b17b7ceaaa848505f706efa32ade43c032df3b988cc0347674ea366358b518e85b62638e5921348fa43d7eec0d94e60043786236393ee9087b22a455265ac872906bddb7666dc8cf4e73a3eda4b6d45a80b5de5d611f583d29860f0cbabddcc2aa869f1ad9be&ascene=1&uin=Mzg0Njg0NzU2&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A0JqCI7pU%2FnjcOVhM6YQQBY%3D&pass_ticket=ooTX65WhM98WGvxntaX7QcZoEQ%2FGQYCZhyRGELfX7aEH4OXRvHNgHqugDy8WUlDP&wx_header=0)写的，这道题挺经典的

栈里放的是柱子的下标，不是高度，是为了求宽度

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int len = heights.length;
        int[] heights_append = new int[len + 2];
        // 首尾添加辅助的0,可以减少特判
        heights_append[0] = 0;
        for(int i = 1; i <= len; i++){
            heights_append[i] = heights[i - 1];
        }
        heights_append[len + 1] = 0;
        int max = 0;
        for(int i = 0; i < len + 2; i++){
            // 注意栈里存的是下标，要从数组里取出高度
            while(!stack.isEmpty() && heights_append[stack.peek()] > heights_append[i]){
                int height = heights_append[stack.pop()];
                int width = i - stack.peek() - 1;
                max = Math.max(max, width * height);
            }
            stack.push(i);

        }
        return max;
    }
}
```



#### 85 最大矩形（84题的变形）

给定一个仅包含 `0` 和 `1` 、大小为 `rows x cols` 的二维二进制矩阵，找出只包含 `1` 的最大矩形，并返回其面积。

思路来自以下题解

https://leetcode-cn.com/problems/maximal-rectangle/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-1-8/

就是把矩阵看成n个柱状图，然后套用上一题的做法就可以了

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        int m = matrix.length;
        if(m == 0) return 0;
        int n = matrix[0].length;
        int[] heights = new int[n + 2];
        int max = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                heights[j + 1] = (matrix[i][j] == '1') ? (heights[j + 1] + 1) : 0;
            }
            max = Math.max(max, maximalRectangle(heights));
        }
        return max;
    }
    int maximalRectangle(int[] heights){
        Stack<Integer> stack = new Stack<>();
        int max = 0;
        for(int i = 0; i < heights.length; i++){
            while(!stack.isEmpty() && heights[stack.peek()] > heights[i]){
                int height = heights[stack.pop()];
                int width = i - stack.peek() - 1;
                max = Math.max(max, height * width);
            }
            stack.push(i);
        }
        return max;
    }
}
```

时间复杂度为O(MN)



#### 86 分隔链表（partition）

给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。

你应当 保留 两个分区中每个节点的初始相对位置。

```
输入：head = [1,4,3,2,5,2], x = 3
输出：[1,2,2,4,3,5]
```

```
输入：head = [2,1], x = 2
输出：[1,2]
```

大致的思路还是比较好想的，就是遍历链表，遇到大于等于x的结点就继续往后，遇到小于x的结点就把它摘下来，放到前面去。因为涉及到删除结点，所以需要一个pre结点来帮助删除，还有插入结点，也需要一个插入的位置。

麻烦的地方是这两个辅助指针的初始值，我定为了dummy head。在遇到示例1时会有问题，简单地说就是当这个结点小于x，但是它不需要删除再插入，它本身的位置不需要改变，的时候，需要用第二个if语句特殊处理，也就是直接跳过它

设置了两个dummy head是因为如果遇到示例1这种情况，dummy head的next就会跳过1

```java
class Solution {
    public ListNode partition(ListNode head, int x) {       
        ListNode dummyHead1 = new ListNode(0, head);
        ListNode dummyHead2 = new ListNode(0, head);
        ListNode iter = dummyHead1;
        ListNode small = dummyHead2;
        while(iter.next != null){
            if(iter.next.val >= x) iter = iter.next;
            
            // 这部分有点难懂，而且这里不能写成iter == small
            // iter.next == samll.next表示要删除的结点已经在它应该在的位置了，不需要删
            else if(iter.next == small.next){
                small = small.next;
                iter.next = iter.next.next;
            }
            
            else{
                ListNode remove = iter.next;
                iter.next = iter.next.next;
                remove.next = small.next;
                small.next = remove;
                small = small.next;
            }
        }
        return dummyHead2.next;
    }
}
```

我又想了一下，还是可以只用一个dummy head的，代码如下

```java
class Solution {
    public ListNode partition(ListNode head, int x) {       
        ListNode dummyHead = new ListNode(0, head);
        ListNode iter = dummyHead;
        ListNode small = dummyHead;
        while(iter.next != null){
            if(iter.next.val >= x) iter = iter.next;
            else if(iter.next == small.next){
                small = small.next;
                iter = iter.next; 
                // 改了这里，iter.next不用删,iter就和前一个if一样往后移就行
                // 这么改之后就不会丢掉结点了
            }
            else{
                ListNode remove = iter.next;
                iter.next = iter.next.next;
                remove.next = small.next;
                small.next = remove;
                small = small.next;
            }
        }
        return dummyHead.next;
    }
}
```

最后去看了题解，啊，题解的思路好简单，我把问题搞麻烦了

> 直观来说我们只需维护两个链表 small 和 large 即可，small 链表按顺序存储所有小于 x 的节点，large 链表按顺序存储所有大于等于 x 的节点。遍历完原链表后，我们只要将 small 链表尾节点指向 large 链表的头节点即能完成对链表的分隔。
> https://leetcode-cn.com/problems/partition-list/solution/fen-ge-lian-biao-by-leetcode-solution-7ade/

```java
class Solution {
    public ListNode partition(ListNode head, int x) {
        ListNode small = new ListNode(0);
        ListNode large = new ListNode(0);
        ListNode smallHead = small;
        ListNode largeHead = large;
        while(head != null){
            if(head.val < x){
                small.next = head;
                small = small.next;
            }
            else{
                large.next = head;
                large = large.next;
            }
            head = head.next;
        }
        small.next = largeHead.next;
        large.next = null;
        return smallHead.next;
    }
}
```



#### 87 扰乱字符串（区间DP，记忆化搜索，递归）

使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：
如果字符串的长度为 1 ，算法停止
如果字符串的长度 > 1 ，执行下述步骤：
在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。
随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。
在 x 和 y 这两个子字符串上继续从步骤 1 开始递归执行此算法。
给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 。

```
输入：s1 = "great", s2 = "rgeat"
输出：true
解释：s1 上可能发生的一种情形是：
"great" --> "gr/eat" // 在一个随机下标处分割得到两个子字符串
"gr/eat" --> "gr/eat" // 随机决定：「保持这两个子字符串的顺序不变」
"gr/eat" --> "g/r / e/at" // 在子字符串上递归执行此算法。两个子字符串分别在随机下标处进行一轮分割
"g/r / e/at" --> "r/g / e/at" // 随机决定：第一组「交换两个子字符串」，第二组「保持这两个子字符串的顺序不变」
"r/g / e/at" --> "r/g / e/ a/t" // 继续递归执行此算法，将 "at" 分割得到 "a/t"
"r/g / e/ a/t" --> "r/g / e/ a/t" // 随机决定：「保持这两个子字符串的顺序不变」
算法终止，结果字符串和 s2 相同，都是 "rgeat"
这是一种能够扰乱 s1 得到 s2 的情形，可以认为 s2 是 s1 的扰乱字符串，返回 true
```

https://leetcode-cn.com/problems/scramble-string/solution/rao-luan-zi-fu-chuan-by-leetcode-solutio-8r9t/

三维DP数组也是没想到了，三个维度分别是s1的子串开始的位置，s2的子串开始的位置，子串的长度

一开始先处理掉完全相同的（直接返回true），再处理掉字母个数不同的（返回false）

然后穷举所有可能的划分，这里需要用递归，为了避免递归中的重复计算，使用了dp数组来记忆计算过的结果

```java
class Solution {
    int[][][] dp;
    String s1;
    String s2;
    public boolean isScramble(String s1, String s2) {
        if(s1.length() != s2.length()) return false;
        this.s1 = s1;
        this.s2 = s2;
        int len = s1.length();
        this.dp = new int[len][len][len + 1];
        return isScramble(0, 0, len);
    }

    public boolean isScramble(int i1, int i2, int length){
        if(dp[i1][i2][length] == 1) return true;
        if(dp[i1][i2][length] == -1) return false;
        
        if(s1.substring(i1, i1 + length).equals(s2.substring(i2, i2 + length))){
            dp[i1][i2][length] = 1;
            return true;
        }

        if(!isSimilar(i1, i2, length)){
            dp[i1][i2][length] = -1;
            return false;
        }
		// 按照切分的长度穷举所有可能
        for(int i = 1; i < length; i++){
            // 不交换
            if(isScramble(i1, i2, i) && isScramble(i1 + i, i2 + i, length - i)){
                dp[i1][i2][length] = 1;
                return true;
            }
            // 交换
            if(isScramble(i1, i2 + length - i, i) && isScramble(i1 + i, i2, length - i)){
                dp[i1][i2][length] = 1;
                return true;
            }
        }
        dp[i1][i2][length] = -1;
        return false;
    }

    public boolean isSimilar(int i1, int i2, int length){
        Map<Character, Integer> map1 = new HashMap<>();
        for(int i = i1; i < i1 + length; i++){
            char c = s1.charAt(i);
            map1.put(c, map1.getOrDefault(c, 0) + 1);
        }
        // Map<Character, Integer> map2 = new HashMap<>();
        // for(int i = i2; i < i2 + length; i++){
        //     map2.put(s2.charAt(i2), map2.getOrDefault(s2.charAt(i2), 0) + 1);
        // }
        // return map1.equals(map2);
        // equals不能起效，不懂
        
        for(int i = i2; i < i2 + length; i++){
            char c = s2.charAt(i);
            map1.put(c, map1.getOrDefault(c, 0) - 1);
        }

        for(Integer value : map1.values()){
            if (value.intValue() != 0) {
                return false;
            }
        }
        return true;
    }
}
```



#### 88 原地合并两个有序数组

给你两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目。

请你 合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 nums1 中。为了应对这种情况，nums1 的初始长度为 m + n，其中前 m 个元素表示应合并的元素，后 n 个元素为 0 ，应忽略。nums2 的长度为 n 。

挺简单的，因为有空位其实好做的，只要倒过来遍历就行了

```java
class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        while(i >= 0 && j >= 0){
            if(nums1[i] >= nums2[j]){
                nums1[k] = nums1[i];
                i--;
            }
            else{
                nums1[k] = nums2[j];
                j--;
            }
            k--;
        }
        while(j >= 0){
            nums1[k] = nums2[j];
            j--;
            k--;
        }
    }
}
```



#### 89 格雷编码（挺有趣的数学题，需要记住规律）

n 位格雷码序列 是一个由 2n 个整数组成的序列，其中：
每个整数都在范围 [0, 2n - 1] 内（含 0 和 2n - 1）
第一个整数是 0
一个整数在序列中出现 不超过一次
每对 相邻 整数的二进制表示 恰好一位不同 ，且
第一个 和 最后一个 整数的二进制表示 恰好一位不同
给你一个整数 n ，返回任一有效的 n 位格雷码序列 。

看了题解才知道原来这么简单

https://leetcode-cn.com/problems/gray-code/solution/gray-code-jing-xiang-fan-she-fa-by-jyd/https://leetcode-cn.com/problems/gray-code/solution/gray-code-jing-xiang-fan-she-fa-by-jyd/

假设有n-1位的格雷编码，要获得n位格雷编码，只要把n-1位的编码按照逆序在最前面加上1，也就是加上2的n-1次方，就可以得到剩下的一半，和前面n-1位的拼起来就是n位的了，前面的没有加1就是最前面是0

```java
class Solution {
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        int head = 1;
        for(int i = 0; i < n; i++){
            for(int j = res.size() - 1; j >= 0; j--){
                res.add(head + res.get(j));
            }
            head <<= 1;
        }
        return res;
    }
}
```



#### 90 子集二（回溯，含重复元素）

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

示例 1：

```
输入：nums = [1,2,2]
输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
```

示例 2：

```
输入：nums = [0]
输出：[[],[0]]
```


提示：

+ 1 <= nums.length <= 10
+ -10 <= nums[i] <= 10

一开始用的是list.contains来判断前面有没有，当前的要不要放。结果没法处理类似[1,1,1]这种情况。然后我就怀疑是list.contains这个方法的问题。后来实验证明了不是它的问题。它是用equals比较的，比如说[1,1,1]，当list里有第一个1的时候，你问他有没有第二个1，会返回true，这时候如果正好在第三个位置，就会以为第二个已经加了，然后把第三个也放进去，其实不行。

所以我用了visited数组。题解用的是preVisited变量，更省

```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        boolean[] visited = new boolean[nums.length];
        dfs(res, list, 0, nums, visited);
        return res;
    }
     void dfs(List<List<Integer>> res, List<Integer> list, int cur, int[] nums, boolean[] visited){
        if(cur == nums.length){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        if(cur > 0 && nums[cur] == nums[cur - 1] && !visited[cur - 1]){}
        else{
            list.add(nums[cur]);
            visited[cur] = true;
            dfs(res, list, cur + 1, nums, visited);
            list.remove(list.size() - 1);
            visited[cur] = false;
        }
        dfs(res, list, cur + 1, nums, visited);
    }
}
```

使用preVisited优化解法

```java
class Solution {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> list = new ArrayList<>();        
        dfs(res, list, 0, nums, false);
        return res;
    }
     void dfs(List<List<Integer>> res, List<Integer> list, int cur, int[] nums, boolean preVisited){
        if(cur == nums.length){
            res.add(new ArrayList<Integer>(list));
            return;
        }
        if(cur > 0 && nums[cur] == nums[cur - 1] && !preVisited){}
        else{
            list.add(nums[cur]);            
            dfs(res, list, cur + 1, nums, true);
            list.remove(list.size() - 1);            
        }
        dfs(res, list, cur + 1, nums, false);
    }
}
```



#### 91 解码方法（一维DP，递推，不能使用深搜）

一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

```
'A' -> 1
'B' -> 2
...
'Z' -> 26
```

要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：

+ "AAJF" ，将消息分组为 (1 1 10 6)
+ "KJF" ，将消息分组为 (11 10 6)

注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。

给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。

**注意**：这道题不能用dfs的写法

一开始自然地想的是dfs的写法，结果超时了，仔细想想，我的dfs会有重复计算的情况

其实表面上的区别就是向前算（递推）和向后算（递归），但是实际上递推才能避免重复计算，并且利用到前面的结果

dp[i]表示的是到第i个字符有几种解码方法

**注意**当没有字符时应该算1种解码方法，dp[0] = 1

```java
class Solution {
    public int numDecodings(String s) {
        if(s.charAt(0) == '0') return 0;
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 1; i < s.length(); i++){
            if(s.charAt(i) != '0'){
                dp[i + 1] += dp[i];
            }
            if(s.charAt(i - 1) != '0' && 10 * (s.charAt(i - 1) - '0') + (s.charAt(i) - '0') < 27 ){
                dp[i + 1] += dp[i - 1];
            }
        }
        return dp[s.length()];
    }
}
```

把dp数组优化成3个变量，代码如下：

```java
class Solution {
    public int numDecodings(String s) {
        if(s.charAt(0) == '0') return 0;
        int first = 1, second = 1, third = 0;
        for(int i = 1; i < s.length(); i++){
            if(s.charAt(i) != '0') third += second;
            if(s.charAt(i - 1) != '0' && 10 * (s.charAt(i - 1) - '0') + (s.charAt(i) - '0') < 27 ){
                third += first;
            }
            first = second;
            second = third;
            third = 0;
        }
        return second;
    }
}
```



#### 92 反转链表二

给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。 你可以使用一趟扫描完成反转吗？

```java
class Solution {
    public ListNode reverseBetween(ListNode head, int left, int right) {
        int index = 0;
        ListNode dummy = new ListNode(0, head);
        ListNode pre = dummy;
        while(index < left - 1){
            pre = pre.next;
            index++;
        }
        ListNode iter = pre.next;
        for(int i = 0; i < right - left; i++){
            ListNode del = iter.next;
            iter.next = del.next;
            del.next = pre.next;
            pre.next = del;
        }
        return dummy.next;
    }
}
```

要考虑几种边界条件，left = 1; right = nums.length; left = right

我用的方法是不断把节点摘下来插入到前面



#### 93 复原IP地址（回溯，不要怕）

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你不能重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。

```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        List<String> list = new ArrayList<>();
        dfs(res, s, 0, list);
        return res;
    }
    void dfs(List<String> res, String s, int index, List<String> list){
        if(index == s.length() && list.size() == 4){
            res.add(String.join(".", list));// 注意这里需要用双引号，否则报错
            return;
        }
        // 这里也需要判断是否不符合条件
        else if(index >= s.length() || list.size() >= 4) return;
		
        // 0比较特殊
        if(s.charAt(index) == '0'){
            list.add("0");
            dfs(res, s, index + 1, list);
            list.remove(list.size() - 1);
        }

        else{
            int sum = 0;
            for(int i = 0; i < 3; i++){
                if(index + i == s.length()) break; 
                // 注意这里要判断是否越界，可能最后剩不到3位了
                sum = sum * 10 + (s.charAt(index + i) - '0');
                if(sum <= 255){
                    list.add(s.substring(index, index + i + 1));
                    dfs(res, s, index + i + 1, list);
                    list.remove(list.size() - 1);
                }
            }
        }
    }
}
```



#### 94 二叉树的中序遍历

**递归**

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderTraversal(root, res);
        return res;
    }
    void inorderTraversal(TreeNode root, List<Integer> list) {
        if(root == null) return;
        inorderTraversal(root.left, list);
        list.add(root.val);
        inorderTraversal(root.right, list);
        return;
    }
}
```

**非递归（来自官方题解）**

```java
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        Deque<TreeNode> stk = new LinkedList<TreeNode>();
        while (root != null || !stk.isEmpty()) {
            while (root != null) {
                stk.push(root);
                root = root.left;
            }
            root = stk.pop();
            res.add(root.val);
            root = root.right;
        }
        return res;
    }
}
```



#### 95 不同的二叉搜索树二（递归）

给你一个整数 `n` ，请你生成并返回所有由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的不同 **二叉搜索树** 。可以按 **任意顺序** 返回答案。

```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        // if(n <= 0) return null;
        return generateTrees(1, n);
    }
    List<TreeNode> generateTrees(int start, int end){
        List<TreeNode> res = new ArrayList<>();
        if(start > end) {
            res.add(null); // 这一步挺重要的
            return res;
        }
        if(start == end){
            TreeNode root = new TreeNode(start);
            res.add(root);
            return res;
        }
        for(int i = start; i <= end; i++){
            List<TreeNode> lefts = generateTrees(start, i - 1);
            List<TreeNode> rights = generateTrees(i + 1, end);
            for(int j = 0; j < lefts.size(); j++){                
                for(int k = 0; k < rights.size(); k++){
                    TreeNode root = new TreeNode(i); // 这一步挺重要的
                    root.left = lefts.get(j);
                    root.right = rights.get(k);
                    res.add(root);
                }
            }
        }
        return res;
    }
}
```



#### 96 不同的二叉搜索树（带记忆的递归）

给你一个整数 `n` ，求恰由 `n` 个节点组成且节点值从 `1` 到 `n` 互不相同的 **二叉搜索树** 有多少种？返回满足题意的二叉搜索树的种数。

用dp数组避免重复计算

dp\[i]表示i个节点能组成几种二叉搜索树，计算的时候只要从1到n设为根节点，计算左右子树有多少种，相乘就得出以当前节点为根节点可以有几棵树

```java
class Solution {
    int[] dp;
    public int numTrees(int n) {
        dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        return count(n);
    }
    int count(int n){
        if(dp[n] != 0) return dp[n];
        int res = 0;
        // 以i为根节点
        for(int i = 1; i <= n; i++){
            // 左子树种数 * 右子树种数
            res += count(i - 1) * count(n - i);
        }
        dp[n] = res;
        return res;
    }
}
```



#### 97 交错字符串（类似编辑距离的二维DP）

给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。

两个字符串 s 和 t 交错 的定义与过程如下，其中每个字符串都会被分割成若干 非空 子字符串：

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
交错 是 s1 + t1 + s2 + t2 + s3 + t3 + ... 或者 t1 + s1 + t2 + s2 + t3 + s3 + ...
提示：a + b 意味着字符串 a 和 b 连接。

首先要知道这道题不能用双指针，因为不能给出指针下一步应该怎么走才是最优的。需要用DP

涉及到的难点就是状态表示。这里dp\[i][j]表示的是s1的前i个字符和s2的前j个字符能否交错组成s3的前i+j个字符，如果dp\[i - 1][j] == true，就考察s1的第i个字符是否等于s3的第i+j个字符，如果dp\[i][j - 1] == true，就考察s2的第j个字符是否等于s3的第i+j个字符

```java
class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
        int len1 = s1.length();
        int len2 = s2.length();
        int len3 = s3.length();
        if((len1 + len2) != len3) return false;
        boolean[][] dp = new boolean[len1 + 1][len2 + 1];
        dp[0][0] = true;
        for(int i = 1; i <= len1; i++){
            if(s1.charAt(i - 1) == s3.charAt(i - 1)) dp[i][0] = true;
            else break;
        }
        for(int j = 1; j <= len2; j++){
            if(s2.charAt(j - 1) == s3.charAt(j - 1)) dp[0][j] = true;
            else break;
        }
        for(int i = 1; i <= len1; i++){
            for(int j = 1; j <= len2; j++){
                if(dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1)) dp[i][j] = true;
                else if(dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1)) dp[i][j] = true;
            }
        }
        return dp[len1][len2] == true;
    }
}
```



#### 98 验证二叉搜索树

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**提示：**

- 树中节点数目范围在`[1, 104]` 内
- `-231 <= Node.val <= 231 - 1`

**法一**：中序遍历判断是不是单调增

代码模式类似中序遍历，只是遍历到之后不是输出来，是和前一个遍历到的比较，然后把当前的记为前一个，再接着遍历，而且返回值是以当前结点为根的树是不是二叉搜索树

```java
class Solution {
    long pre = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
        if(root == null) return true;
        if(!isValidBST(root.left)) return false;
        if(root.val <= pre) return false;
        pre = root.val;
        return isValidBST(root.right);
    }
}
```

**法二**：范围比较

看的题解的做法，简单地说就是给每个结点为根的树都设置一个范围

```java
class Solution {
    public boolean isValidBST(TreeNode root){
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    boolean isValidBST(TreeNode root, long low, long high){
        if(root == null) return true;
        if(root.val <= low || root.val >= high) return false;
        if(!isValidBST(root.left, low, root.val)) return false;
        return isValidBST(root.right, root.val, high);
    }
}
```



#### 99 恢复二叉搜索树（有一点点难的中序遍历）

给你二叉搜索树的根节点 `root` ，该树中的 恰好 两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树。

一开始没什么思路，后来看了题解，大致思路是，中序遍历会得到一个序列，两个结点被交换后得到的序列就不是单调增的，可能是12435这种，也可能是15342这种，第一种就是找到4>3，交换4和3，第二种就是找到5>3的5和4>2的2，交换5和2，不需要把整个中序遍历的结果存下来，只要存前一个结点的value和当前结点进行比较就行，然后还有两个node记录需要交换的结点

我的代码如下

```java
class Solution {
    TreeNode pre = null;
    TreeNode node1 = null;
    TreeNode node1Next = null;
    TreeNode node2 = null;
    public void recoverTree(TreeNode root){
        inorder(root);
        if(node2 == null){
            int tmp = node1.val;
            node1.val = node1Next.val;
            node1Next.val = tmp;
        }
        else{
            int tmp = node1.val;
            node1.val = node2.val;
            node2.val = tmp;
        }
        return;
    }
    public void inorder(TreeNode root) {
        if(root == null) return;
        inorder(root.left);
        if(pre != null && pre.val > root.val){
            if(node1 == null){
                node1 = pre;
                node1Next = root;
            }
            else{
                node2 = root;
            }
        }
        pre = root;
        inorder(root.right);
    }
}
```

**进阶：**使用 `O(n)` 空间复杂度的解法很容易实现。你能想出一个只使用 `O(1)` 空间的解决方案吗？





#### 100 相同的二叉树

给你两棵二叉树的根节点 `p` 和 `q` ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null || q == null){
            if(p == null && q == null) return true;
            else return false;
        }
        if(p.val == q.val){
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        }
        else{
            return false;
        }
    }
}
```



#### 一些见解

+ 如果有一个二维数组，且只能向右向下走，很大概率就是二维DP，dp\[i][j]表示的是到(i,j)这个点的结果
+ 字符串编辑距离，正则匹配，最长公共子序列，二维DP
+ 在树/图中求最短距离，BFS
+ 排列组合，回溯，含重复元素就先排序再用visited数组

