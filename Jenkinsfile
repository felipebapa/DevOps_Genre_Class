pipeline {
    agent any

    parameters {
        string(
            name: 'MODEL_ALIAS',
            defaultValue: 'champion',
            description: 'Alias del Model Registry cuya versión se va a desplegar.'
        )
        string(
            name: 'MIN_F1_MACRO',
            defaultValue: '0.30',
            description: 'Gate de calidad: el build falla si el f1_macro del modelo es menor.'
        )
        booleanParam(
            name: 'PUSH_IMAGE',
            defaultValue: true,
            description: 'Desmarcar para construir sin publicar en Docker Hub.'
        )
    }

    environment {
        IMAGE_NAME          = 'leuribe2/devops-genre-class'

        // ── MLflow ─────────────────────────────────────────────────────────
        MODEL_NAME          = 'genre-classifier'
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
        MLFLOW_NETWORK      = 'mlops'
        MLFLOW_IMAGE        = 'genre-mlops/mlflow:3.15.1'

        // Cómo se monta el workspace en los contenedores efímeros que hablan
        // con MLflow. El daemon de Docker es el del host, así que la ruta tiene
        // que existir para él.
        //
        // Jenkins corre NATIVO en hyoga (verificado con Jenkinsfile.infra), de
        // modo que el daemon ve las mismas rutas que el agente y basta montar
        // el workspace sobre sí mismo. Si algún día Jenkins pasara a correr en
        // contenedor, cambiar por el volumen nombrado que use su jenkins_home:
        //   WORKSPACE_MOUNT = 'genre-mlops_jenkins-home:/var/jenkins_home'
        //
        // Ojo: el valor NO puede ser el literal '${WORKSPACE}:${WORKSPACE}'.
        // El shell no expande dos veces, así que Docker recibiría esa cadena
        // sin resolver. Por eso se interpola aquí, en Groovy.
        WORKSPACE_MOUNT     = "${WORKSPACE}:${WORKSPACE}"
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Verify Toolchain') {
            steps {
                sh '''
                    echo "Checking Docker environment..."
                    docker --version
                    test -f Dockerfile

                    # Si la interpolación de ${WORKSPACE} fallara, Docker daría
                    # un error de bind mount difícil de leer tres etapas después.
                    echo "Workspace mount: ${WORKSPACE_MOUNT}"
                    case "${WORKSPACE_MOUNT}" in
                        *'$'*|:*|*:)
                            echo "ERROR: WORKSPACE_MOUNT quedó sin resolver."
                            echo "       Valor actual: '${WORKSPACE_MOUNT}'"
                            echo "       Revisa esa variable en el bloque environment."
                            exit 1
                            ;;
                    esac

                    echo "Ensuring MLflow client image..."
                    docker image inspect ${MLFLOW_IMAGE} >/dev/null 2>&1 || \
                        docker build -t ${MLFLOW_IMAGE} docker/mlflow

                    echo "Checking MLflow tracking server at ${MLFLOW_TRACKING_URI}..."
                    docker run --rm --network ${MLFLOW_NETWORK} ${MLFLOW_IMAGE} \
                        python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('${MLFLOW_TRACKING_URI}/health', timeout=15).status == 200 else 1)"
                    echo "MLflow OK"
                '''
            }
        }

        stage('Fetch Model from Registry') {
            steps {
                script {
                    def status = sh(
                        returnStatus: true,
                        script: '''
                            rm -rf models
                            mkdir -p models

                            docker run --rm \
                                --network ${MLFLOW_NETWORK} \
                                -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
                                -v "${WORKSPACE_MOUNT}" \
                                -w "${WORKSPACE}" \
                                ${MLFLOW_IMAGE} \
                                python scripts/mlflow_fetch_model.py \
                                    --model-name "${MODEL_NAME}" \
                                    --alias "${MODEL_ALIAS}" \
                                    --dest models \
                                    --min-f1-macro "${MIN_F1_MACRO}"
                        '''
                    )

                    if (status == 2) {
                        error("El modelo ${env.MODEL_NAME}@${params.MODEL_ALIAS} no pasa el gate de calidad (f1_macro < ${params.MIN_F1_MACRO}). No se despliega.")
                    } else if (status != 0) {
                        error("No se pudo traer el modelo desde MLflow (exit ${status}). Revisa que el servidor esté arriba y que exista el alias @${params.MODEL_ALIAS}.")
                    }

                    env.MODEL_VERSION = sh(
                        returnStdout: true,
                        script: 'grep "^MODEL_VERSION=" models/MODEL_VERSION.env | cut -d= -f2'
                    ).trim()

                    env.MODEL_RUN_ID = sh(
                        returnStdout: true,
                        script: 'grep "^MODEL_RUN_ID=" models/MODEL_VERSION.env | cut -d= -f2'
                    ).trim()

                    env.MODEL_F1_MACRO = sh(
                        returnStdout: true,
                        script: 'grep "^MODEL_F1_MACRO=" models/MODEL_VERSION.env | cut -d= -f2'
                    ).trim()

                    currentBuild.description =
                        "model v${env.MODEL_VERSION} · f1_macro ${env.MODEL_F1_MACRO}"

                    echo "Desplegando ${env.MODEL_NAME} v${env.MODEL_VERSION} (run ${env.MODEL_RUN_ID})"
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    echo "Building Docker image with model v${MODEL_VERSION}..."
                    docker build \
                        -t ${IMAGE_NAME}:${BUILD_NUMBER} \
                        -t ${IMAGE_NAME}:model-v${MODEL_VERSION} \
                        --label org.opencontainers.image.revision="${GIT_COMMIT}" \
                        --label ai.mlflow.model.name="${MODEL_NAME}" \
                        --label ai.mlflow.model.version="${MODEL_VERSION}" \
                        --label ai.mlflow.run_id="${MODEL_RUN_ID}" \
                        --label ai.mlflow.f1_macro="${MODEL_F1_MACRO}" \
                        .
                '''
            }
        }

        stage('Push Docker Image') {
            when {
                expression { return params.PUSH_IMAGE }
            }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKERHUB_USER',
                        passwordVariable: 'DOCKERHUB_TOKEN'
                    )
                ]) {
                    sh '''
                        echo "$DOCKERHUB_TOKEN" | \
                            docker login \
                            -u "$DOCKERHUB_USER" \
                            --password-stdin

                        docker push ${IMAGE_NAME}:${BUILD_NUMBER}
                        docker push ${IMAGE_NAME}:model-v${MODEL_VERSION}

                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            // Sin push no hay imagen en el registry que Kubernetes pueda bajar.
            when {
                expression { return params.PUSH_IMAGE }
            }
            steps {
                withCredentials([
                    file(
                        credentialsId: 'koga-kubeconfig',
                        variable: 'KUBECONFIG'
                    )
                ]) {
                    sh '''
                        echo "Deploying build ${BUILD_NUMBER} to Koga..."

                        kubectl set image \
                            deployment/genre-classifier \
                            genre-classifier=${IMAGE_NAME}:${BUILD_NUMBER}

                        kubectl rollout status \
                            deployment/genre-classifier \
                            --timeout=600s
                    '''
                }
            }
        }

        // Va después del rollout: así la versión solo queda marcada como
        // desplegada si Kubernetes efectivamente la puso a correr.
        stage('Record Deployment in MLflow') {
            when {
                expression { return params.PUSH_IMAGE }
            }
            steps {
                sh '''
                    docker run --rm \
                        --network ${MLFLOW_NETWORK} \
                        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
                        -v "${WORKSPACE_MOUNT}" \
                        -w "${WORKSPACE}" \
                        ${MLFLOW_IMAGE} \
                        python scripts/mlflow_tag_deployment.py \
                            --model-name "${MODEL_NAME}" \
                            --version "${MODEL_VERSION}" \
                            --image "${IMAGE_NAME}:${BUILD_NUMBER}" \
                            --build-number "${BUILD_NUMBER}" \
                            --build-url "${BUILD_URL}" \
                            --git-commit "${GIT_COMMIT}"
                '''
            }
        }
    }

    post {
        always {
            sh '''
                echo "Cleaning local Jenkins images..."
                docker image rm ${IMAGE_NAME}:${BUILD_NUMBER} || true
                if [ -n "${MODEL_VERSION}" ]; then
                    docker image rm ${IMAGE_NAME}:model-v${MODEL_VERSION} || true
                fi

                echo "Removing model artifacts fetched from the registry..."
                rm -rf models

                # Al quitar los tags de arriba, las capas de la imagen quedan
                # huérfanas y siguen ocupando disco. hyoga va justa de espacio y
                # cada build genera una imagen de varios GB, así que se liberan.
                # Solo `image prune` (sin -a): borra lo que ya no referencia
                # ningún tag, nunca imágenes en uso como el cliente de MLflow.
                echo "Reclaiming dangling image layers..."
                docker image prune -f || true

                echo "Disk usage after cleanup:"
                docker system df || true
            '''
        }
        success {
            echo "OK: imagen ${env.IMAGE_NAME}:${env.BUILD_NUMBER} publicada con ${env.MODEL_NAME} v${env.MODEL_VERSION}"
        }
    }
}
