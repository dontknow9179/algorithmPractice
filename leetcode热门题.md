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



#### 3 无重复字符的最长子串（经典，双指针，滑动窗口）

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
> 在确定第一个数之后，如果nums[i] + nums[len - 3] + nums[len - 2] + nums[len - 1] < target，说明此时剩下的三个数无论取什么值，四数之和一定小于target，因此第一重循环直接进入下一轮，枚举nums[i+1]；
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

参考题解写的单调栈（递减）的代码如下：

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



#### 56 合并区间

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
                    maps.put(del, maps.get(del) - 1);
                    if(maps.get(del) < mapt.get(del)) count--;
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
>

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
>

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
            res += map.getOrDefault(sum - k, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return res;
    }
}
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



#### 752 打开转盘锁

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

