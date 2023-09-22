"selfsigned-certificate": {
    alias: ""
    annotations: {}
    attributes: workload: type: "componentdefinitions.core.oam.dev"
    description: "self signed certificates"
    labels: {}
    type: "component"
}

template: {
    output: {
        kind: "Issuer"
        apiVersion: "cert-manager.io/v1"
        metadata: {
            name: "selfsigned"
            namespace: parameter.namespace
        }
        spec: {
            selfSigned: {}
        }
    }
    outputs: {
        certificate: {
            kind: "Certificate"
            apiVersion: "cert-manager.io/v1"
            metadata: {
                name: "selfsigned"
                namespace: parameter.namespace
            }
            spec: {
                commonName: "my-certificate"
                secretName: "test-secret"
                issuerRef:
                    name: "selfsigned"
                    kind: "Issuer"
                    namespace: parameter.namespace
                    group: "cert-manager.io"
            }
        }
    }

    parameter: {
        namespace: *"default" |string
    }
}

