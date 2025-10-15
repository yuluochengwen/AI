#include <stdio.h>

int main(){
    int var = 10;
    int *p;
    p = &var;
    printf("var的地址：%p\n", p);
    printf("通过p访问var的值：%d", *p);
    return 0;
}