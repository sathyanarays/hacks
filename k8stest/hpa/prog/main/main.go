package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	fmt.Println("HelloWorld")
	i, _ := strconv.Atoi(os.Args[1])
	arr := []int{}
	done := make(chan bool)
	go func() {
		for j := 0; j < i*1024*1024; j++ {
			arr = append(arr, 0)
		}
		done <- true
	}()

	<-done
	fmt.Println(len(arr))
}

// 2    => 1831784K
// 4    => 1905260K
// 1024 => 2052468K
