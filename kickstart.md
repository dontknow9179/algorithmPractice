### kickstart

#### 找数组中大于前面所有数且大于后一个数的数字有几个

```java
import java.util.*;
import java.io.*;

public class Solution{
    public static void main(String[] args){
        Solution m = new Solution();
        m.func();
    }
    public void func(){
        Scanner sc = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = sc.nextInt();
        int n, max, y;
        for(int x = 1; x <= t; x++){
            n = sc.nextInt();
            
            y = 0;
            int[] nums = new int[n];
            for(int i = 0; i < n; i++){
                nums[i] = sc.nextInt();
            }
            max = nums[0];
            if(n > 1 && nums[0] > nums[1]){
                y++;
            }
            if(n == 1) y = 1;
            for(int i = 1; i < n; i++){
                if(nums[i] > max && (i == n - 1 || nums[i] > nums[i + 1])){
                    y++;
                }
                max = Math.max(max, nums[i]);
            }
            System.out.println("Case #" + x + ": " + y);
        }
    }
}
```

+ 当数组中只有一个数时要返回1

```java
			max = -1;

            for(int i = 0; i < n; i++){
                if(nums[i] > max && (i == n - 1 || nums[i] > nums[i + 1])){
                    y++;
                }
                max = Math.max(max, nums[i]);
            }
```

