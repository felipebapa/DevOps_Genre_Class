pipeline {
    agent any

    environment {
        IMAGE_NAME = 'leuribe2/devops-genre-class'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Verify Docker') {
            steps {
                sh '''
                    echo "Checking Docker environment..."
                    docker --version
                    test -f Dockerfile
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    echo "Building Docker image..."
                    docker build \
                        -t ${IMAGE_NAME}:${BUILD_NUMBER} \
                        .
                '''
            }
        }

        stage('Push Docker Image') {
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

                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
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
    }

    post {
        always {
            sh '''
                echo "Cleaning local Jenkins image..."
                docker image rm ${IMAGE_NAME}:${BUILD_NUMBER} || true
            '''
        }
    }
}
