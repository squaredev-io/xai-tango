pipeline {
    agent {
        node {
            label "Agent01" // Replace with your Jenkins agent label
        }
    }

    tools {
        maven "Maven-v.3.6.3" // Replace with the required Maven version
        jdk "jdk11.0"         // Replace with the required JDK version
    }

    environment {
        STREAMLIT_APP_NAME = "xai-streamlit"
        API_APP_NAME = "xai-api"
        DOCKER_REGISTRY = "harbor.tango.rid-intrasoft.eu"
        STREAMLIT_IMAGE_TAG = "${DOCKER_REGISTRY}/xai/${STREAMLIT_APP_NAME}:${BUILD_NUMBER}"
        API_IMAGE_TAG = "${DOCKER_REGISTRY}/xai/${API_APP_NAME}:${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out source code...'
                checkout scm
            }
        }

        stage('Build Streamlit Image') {
            steps {
                script {
                    echo 'Building Streamlit Docker image...'
                    sh "docker build -t ${STREAMLIT_IMAGE_TAG} . -f streamlit.Dockerfile"
                }
            }
        }

        stage('Build API Image') {
            steps {
                script {
                    echo 'Building API Docker image...'
                    sh "docker build -t ${API_IMAGE_TAG} . -f api.Dockerfile"
                }
            }
        }

        stage('Push Streamlit Image') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'harbor-jenkins-creds', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
                        echo 'Pushing Streamlit Docker image...'
                        sh "docker login ${DOCKER_REGISTRY} -u ${USERNAME} -p ${PASSWORD}"
                        sh "docker push ${STREAMLIT_IMAGE_TAG}"
                        sh "docker tag ${STREAMLIT_IMAGE_TAG} ${DOCKER_REGISTRY}/xai/${STREAMLIT_APP_NAME}:latest"
                        sh "docker push ${DOCKER_REGISTRY}/xai/${STREAMLIT_APP_NAME}:latest"
                    }
                }
            }
        }

        stage('Push API Image') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'harbor-jenkins-creds', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
                        echo 'Pushing API Docker image...'
                        sh "docker login ${DOCKER_REGISTRY} -u ${USERNAME} -p ${PASSWORD}"
                        sh "docker push ${API_IMAGE_TAG}"
                        sh "docker tag ${API_IMAGE_TAG} ${DOCKER_REGISTRY}/xai/${API_APP_NAME}:latest"
                        sh "docker push ${DOCKER_REGISTRY}/xai/${API_APP_NAME}:latest"
                    }
                }
            }
        }

        stage('Clean Up Local Images') {
            steps {
                script {
                    echo 'Cleaning up local Docker images...'
                    sh "docker rmi ${STREAMLIT_IMAGE_TAG}"
                    sh "docker rmi ${DOCKER_REGISTRY}/xai/${STREAMLIT_APP_NAME}:latest"
                    sh "docker rmi ${API_IMAGE_TAG}"
                    sh "docker rmi ${DOCKER_REGISTRY}/xai/${API_APP_NAME}:latest"
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                script {
                    withKubeConfig([credentialsId: 'K8s-config-file', serverUrl: 'https://k8s-cluster-url:6443', namespace: 'tango-development']) {
                        echo 'Deploying Streamlit and API apps to Kubernetes...'
                        sh 'kubectl apply -f tango-infrastructure/manifests/xai-tools.yml'
                        sh 'kubectl apply -f tango-infrastructure/manifests/xai-tools-ingress.yml'
                        sh 'kubectl get pods'
                    }
                }
            }
        }
    }

    post {
        failure {
            echo 'Build or deployment failed!'
            slackSend(color: "#FF0000", message: "Job FAILED: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
        }
        success {
            echo 'Build and deployment succeeded!'
            slackSend(color: "#008000", message: "Job SUCCESSFUL: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
        }
    }
}
