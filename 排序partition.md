## 快速选择

```java
private static int partition(int[] list, int low, int high){
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
        list[low] = tmp;//or list[high] = tmp;
        return low;//or return high;
}

public int findKthLargest(int[] nums, int k) {
    k = nums.length - k;
    int l = 0, h = nums.length - 1;
    while (l < h) {
        int j = partition(nums, l, h);
        if (j == k) {
            break;
        } else if (j < k) {
            l = j + 1;
        } else {
            h = j - 1;
        }
    }
    return nums[k];
}
```



### 荷兰国旗问题

```
Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```



```java
public void sortColors(int[] nums) {
    int zero = -1, one = 0, two = nums.length;
    while (one < two) {
        if (nums[one] == 0) {
            swap(nums, ++zero, one++);
        } else if (nums[one] == 2) {
            swap(nums, --two, one);
        } else {
            ++one;
        }
    }
}

private void swap(int[] nums, int i, int j) {
    int t = nums[i];
    nums[i] = nums[j];
    nums[j] = t;
}
```

