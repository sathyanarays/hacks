package main

import (
	"bytes"
	"fmt"
	"text/template"
)

func main() {
	var b bytes.Buffer
	letter := "Hello {{ .Spec.Environment }} !!!"
	tmp := template.Must(template.New("letter").Parse(letter))
	obj := Obj{
		Spec: Spec{
			Environment: "prod",
		},
	}
	e := tmp.Execute(&b, obj)
	if e != nil {
		fmt.Println("Error", e)
	}

	fmt.Println(string(b.String()))

	obj1 := map[string]interface{}{
		"Spec": map[string]interface{}{
			"Environment": "non-prod",
		},
	}

	var b1 bytes.Buffer
	e = tmp.Execute(&b1, obj1)
	if e != nil {
		fmt.Println("Error", e)
	}

	fmt.Println(string(b1.String()))

}

type Obj struct {
	Spec Spec
}

type Spec struct {
	Environment string
}
