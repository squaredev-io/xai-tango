pipeline {
  agent {
    node {
      label "Agent01"
    }
  }

  tools {
    maven "Maven-v.3.6.3"
    jdk "jdk11.0"
  }

  environment {
    STREAMLIT_APP_NAME = "xai-tools"
    API_APP_NAME = "xai-api"
    ARTIFACTORY_SERVER = "harbor.tango.rid-intrasoft.eu"
    ARTIFACTORY_DOCKER_REGISTRY = "harbor.tango.rid-intrasoft.eu/xai/"
    BRANCH_NAME = "main"
    STREAMLIT_IMAGE_TAG = "$STREAMLIT_APP_NAME:R${env.BUILD_ID}"
    API_IMAGE_TAG = "$API_APP_NAME:R${env.BUILD_ID}"
  }

  stages {
    stage('Checkout') {
      steps {
        echo 'Checkout SCM'
        checkout scm
        checkout([$class: 'GitSCM',
                  branches: [[name: env.BRANCH_NAME]],
                  extensions: [[$class: 'CleanBeforeCheckout']],
                  userRemoteConfigs: scm.userRemoteConfigs
        ])
      }
    }

    stage('Build Streamlit Image') {
      steps {
        echo 'Building Streamlit Docker Image'
        script {
          docker.build("${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_IMAGE_TAG}", "-f streamlit.Dockerfile .")
        }
      }
    }

    stage('Build API Image') {
      steps {
        echo 'Building API Docker Image'
        script {
          docker.build("${ARTIFACTORY_DOCKER_REGISTRY}${API_IMAGE_TAG}", "-f api.Dockerfile .")
        }
      }
    }

    stage("Push Images") {
      steps {
        withCredentials([[$class: 'UsernamePasswordMultiBinding', credentialsId: 'harbor-jenkins-creds', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD']]) {
          echo "***** Pushing Docker Images *****"
          sh 'docker login ${ARTIFACTORY_SERVER} -u ${USERNAME} -p ${PASSWORD}'
          sh 'docker image push ${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_IMAGE_TAG}'
          sh 'docker image push ${ARTIFACTORY_DOCKER_REGISTRY}${API_IMAGE_TAG}'
          sh 'docker tag ${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_IMAGE_TAG} ${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_APP_NAME}:latest_dev'
          sh 'docker tag ${ARTIFACTORY_DOCKER_REGISTRY}${API_IMAGE_TAG} ${ARTIFACTORY_DOCKER_REGISTRY}${API_APP_NAME}:latest_dev'
          sh 'docker image push ${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_APP_NAME}:latest_dev'
          sh 'docker image push ${ARTIFACTORY_DOCKER_REGISTRY}${API_APP_NAME}:latest_dev'
        }
      }
    }

    stage('Remove Local Docker Images') {
      steps {
        sh 'docker rmi "${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_IMAGE_TAG}"'
        sh 'docker rmi "${ARTIFACTORY_DOCKER_REGISTRY}${API_IMAGE_TAG}"'
        sh 'docker rmi "${ARTIFACTORY_DOCKER_REGISTRY}${STREAMLIT_APP_NAME}:latest_dev"'
        sh 'docker rmi "${ARTIFACTORY_DOCKER_REGISTRY}${API_APP_NAME}:latest_dev"'
      }
    }

    stage("Deploy Applications") {
      steps {
        withKubeConfig([credentialsId: 'K8s-config-file', serverUrl: 'https://167.235.66.115:6443', namespace: 'tango-development']) {
          echo "Deploying Streamlit and API to Kubernetes"
          sh 'kubectl apply -f tango-infrastructure/manifests/xai-tools.yml'
          sh 'kubectl apply -f tango-infrastructure/manifests/xai-tools-ingress.yml'
          sh 'kubectl get pods -n tango-development'
        }
      }
    }
  }

  post {
    failure {
      slackSend(color: "#FF0000", message: "Job FAILED: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
    }
    success {
      slackSend(color: "#008000", message: "Job SUCCESSFUL: '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL})")
    }
  }
}
