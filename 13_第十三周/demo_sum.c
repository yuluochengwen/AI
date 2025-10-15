#include <stdio.h>//头文件
#define PI 3.1415926
//函数声明
int add(int a, int b);

//函数定义
int add(int a, int b){
    return a + b;
}

int main(){
    //变量声明
    int num1, num2, sum;
    //输入
    printf("请输入两个整数：");
    scanf("%d %d", &num1, &num2);
    //调用函数
    sum = add(num1, num2);
    //输出
    printf("两个整数的和为：%d\n", sum);
    return 0;//返回0表示程序成功执行
}