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

