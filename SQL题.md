## SQL题

#### 175 组合两个表

编写一个 SQL 查询，满足条件：无论 person 是否有地址信息，都需要基于上述两表提供 person 的以下信息：

```
FirstName, LastName, City, State
```

考察的是left join

```sql
select firstname, lastname, city, state from person left join address on person.personid = address.personid
```



#### 176 第二高的薪水

要求没有第二高的时候返回null，而不是空，看了题解使用了临时表

第k高用了order by xxx desc limit 1 offset k-1

offset 表示跳过前面的几个

```sql
select 
(select distinct 
 	salary 
 from 
 	employee 
 order by salary desc 
 limit 1 offset 1) as SecondHighestSalary
```

另一种做法是利用ifnull函数

```sql
SELECT
    IFNULL(
      (SELECT DISTINCT Salary
       FROM Employee
       ORDER BY Salary DESC
        LIMIT 1 OFFSET 1),
    NULL) AS SecondHighestSalary
```



#### 177 第N高的薪水

编写一个SQL查询来报告 `Employee` 表中第 `n` 高的工资。如果没有第 `n` 个最高工资，查询应该报告为 `null` 。

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    SET N := N - 1; # 要放在return外面
    RETURN (
      # Write your MySQL query statement below.    
        select 
            (select distinct 
 	            salary 
            from 
 	            employee 
            order by salary desc 
            limit 1 offset N) 
    );
END
```



#### 178 分数排名（4大排名函数）

编写 SQL 查询对分数进行排序。排名按以下规则计算:

分数应按从高到低排列。
如果两个分数相等，那么两个分数的排名应该相同。
在排名相同的分数后，排名数应该是下一个连续的整数。换句话说，排名之间不应该有空缺的数字。
按 score 降序返回结果表。

+ rank()

+ dense_rank()
+ row_number() 在排名时序号 连续 不重复，即使遇到表中的两个一样的数值亦是如此
+ ntil(group_count)

```sql
dense_rank() over(order by score desc)
```

答案

```sql
select score, dense_rank() over(order by score desc) as "rank" from scores;
```

