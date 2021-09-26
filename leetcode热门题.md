### leetcode热门题

#### 1 两数之和（利用哈希表加速）

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** *`target`* 的那 **两个** 整数，并返回它们的数组下标。

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



#### 41 缺失的第一个正数（Hard, 桶排序, 因为swap找了半天bug）

给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。 

进阶：你可以实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案吗？

从[宫水三叶的刷题日记](https://mp.weixin.qq.com/s?__biz=MzU4NDE3MTEyMA==&mid=2247486339&idx=1&sn=9b351c45a2fe1666ea2905dccbc11d0c&chksm=fd9ca09ccaeb298a4dfcc071911e9e99594befdc5f62860e7b602d11fc862d6bddc43f704ecc&token=1087019265&lang=zh_CN&scene=21#wechat_redirect)看的两种思路

第一种是先排序，然后查找1到n的数哪一个不在数组中。思路简单，复杂度为nlogn，由于n<=300，log300<10可以视为常数

第二种是桶排序（看不太懂为啥叫做桶排序，感觉不太像）

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

