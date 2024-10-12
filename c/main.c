#include <stdio.h>
#include "foo.h"

// https://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html
int main(void)
{
    puts("This is a shared library test");
    foo();
    return 0;
}