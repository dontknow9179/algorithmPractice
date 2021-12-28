### leetcode热门题

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



#### 3 无重复字符的最长子串（经典，双指针）

给定一个字符串 `s` ，请你找出其中不含有重复字符的 **最长子串** 的长度。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if(s.length() == 0) return 0;
        if(s.length() == 1) return 1;
        int max = 1;
        int left = 0, right = 1;
        HashSet<Character> set = new HashSet<>();
        set.add(s.charAt(left));
        while(left <= right && right < s.length()){
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

在题解里找了一下不用开新数组的写法，如下：

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

从高位到地位模拟，说实话**很傻**，如果数字范围再大点就不能这么搞了

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
> 在确定第一个数之后，如果nums[i] + nums[j] + nums[len - 2] + nums[len - 1] < target，说明此时剩下的三个数无论取什么值，四数之和一定小于target，因此第一重循环直接进入下一轮，枚举nums[i+1]；
> 在确定前两个数之后，如果nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target，说明此时剩下的两个数无论取什么值，四数之和一定大于target，因此退出第二重循环；
> 在确定前两个数之后，如果nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target，说明此时剩下的两个数无论取什么值，四数之和一定小于 target，因此第二重循环直接进入下一轮，枚举nums[j+1]。



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

#### 28

#### 29

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
+ 我的代码里dp[0]是填充的，和上面的式子会有点不一样

```java
class Solution {
    public int longestValidParentheses(String s) {
        int len = s.length();
        int[] dp = new int[len + 1];
        int max = 0;
        for(int i = 0; i < len; i++){
            if(s.charAt(i) == '('){
                dp[i + 1] = 0;
            }
            else if(i > 0){
                if(s.charAt(i - 1) == '('){
                    dp[i + 1] = dp[i - 1] + 2;
                }
                if(s.charAt(i - 1) == ')' && i - dp[i] - 1 >= 0 && s.charAt(i - dp[i] - 1) == '('){
                    dp[i + 1] = dp[i] + 2;
                    if(i - dp[i] - 2 >= 0){
                        dp[i + 1] += dp[i - dp[i] - 1];
                    } 
                }
                max = Math.max(max, dp[i + 1]);
            }
        }
        return max;
    }
}
```



#### 33 搜索旋转排序数组（二分查找经典变体）

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
                if(solveSudoku(board, pos + 1)) return true;
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
            while(i > 0 && candidates[i] == candidates[i - 1] && !visited[i - 1]) {
                i++;
                if(i >= candidates.length) break;
            }
            if(i >= candidates.length) break;
            if(candidates[i] <= target){
                list.add(candidates[i]);
                visited[i] = true;
                dfs(candidates, i + 1, target - candidates[i], res, list, visited);
                list.remove(list.size() - 1);
                visited[i] = false;
            }
        }
        return;
    }
}
```



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

参考题解写的单调栈的代码如下：

```java
class Solution {
    public int trap(int[] height) {
        if(height.length == 0) return 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int res = 0;
        for(int i = 1; i < height.length; i++){
            while(!stack.isEmpty() && height[i] > height[stack.peek()]){
                int top = stack.pop();
                if(stack.isEmpty()) break;
                int distance = i - stack.peek() - 1;
                int high = Math.min(height[i], height[stack.peek()]) - height[top];
                res += high * distance;
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



#### 47 全排列二（类似第39题）

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



#### 84 柱状图中最大的矩形（单调栈）

给定 *n* 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

根据[五分钟学算法](https://mp.weixin.qq.com/s?__biz=MzUyNjQxNjYyMg==&mid=2247498521&idx=2&sn=07e4126177f09aae483dbbbf3c64cde1&chksm=fa0d9498cd7a1d8e457d9e140eb7d31c4354f8380a070d6aa4df7839b05e21022e1e784d3db1&scene=126&sessionid=1616597425&key=4b3245558f41098e4b8a077ad5a361c77e7ab4dbd7f405cd9534b17b7ceaaa848505f706efa32ade43c032df3b988cc0347674ea366358b518e85b62638e5921348fa43d7eec0d94e60043786236393ee9087b22a455265ac872906bddb7666dc8cf4e73a3eda4b6d45a80b5de5d611f583d29860f0cbabddcc2aa869f1ad9be&ascene=1&uin=Mzg0Njg0NzU2&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A0JqCI7pU%2FnjcOVhM6YQQBY%3D&pass_ticket=ooTX65WhM98WGvxntaX7QcZoEQ%2FGQYCZhyRGELfX7aEH4OXRvHNgHqugDy8WUlDP&wx_header=0)写的，这道题挺经典的

```java
class Solution {
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int len = heights.length;
        int[] heights_append = new int[len + 2];
        heights_append[0] = 0;
        for(int i = 1; i <= len; i++){
            heights_append[i] = heights[i - 1];
        }
        heights_append[len + 1] = 0;
        int max = 0;
        for(int i = 0; i < len + 2; i++){
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



#### 215 数组中的第K个最大元素（面试高频题）

给定整数数组 `nums` 和整数 `k`，请返回数组中第 `**k**` 个最大的元素。

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

参考了[五分钟学算法](https://mp.weixin.qq.com/s?__biz=MzUyNjQxNjYyMg==&mid=2247498838&idx=2&sn=c10692783ae0bee1312ffe1aa7d10339&chksm=fa0d93d7cd7a1ac1ceff349bfe2a67770bdcebca3bd8a13c5148e6aacab087dbee8154cbeea8&scene=126&sessionid=1616233063&key=4764af25be59e44a2a6d0a22e84b56ef4f29cd5450586618381eda3dcc1a89ac4bc5c790ef342db8ed6bb61acafba342aa10f2ddac5900b381931c725a65f0db410c00813cad0ed142edc3df638b4c586d9a073685ba6c1f02a63c1f9e7c3e36cc53a2217df5d370a5a423bc947a7107213ecd9792c609729ee34dc8950465d8&ascene=1&uin=Mzg0Njg0NzU2&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A%2BQKkwMXJve5HGlpk1VN8u0%3D&pass_ticket=R%2F0%2Fb2w4jP8VfyFlSRsubdTmnhgWNUHpUDtMa%2FGe957YH6%2BYbTc3O%2FLJjZZMGFnF&wx_header=0)里的图解写出下面的代码

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length;
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

**注意**和剑指offer的59-1相同，但是代码复制过去却不行。

**原因** 剑指里输入的数组可能长度为0，需要增加一个判断，长度为零时返回[]



#### 450 删除二叉搜索树中的节点（掌握模板）

给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

首先找到需要删除的节点；
如果找到了，删除它。

看了官方题解做的，一开始真的毫无思路

首先要确定找到要删的结点之后怎么删，这是最重要的！

+ 要拿它的前驱或者后继节点来替代它
+ 要怎么替代它？之间替吗？
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

