def myflag = false

pipeline {
  options {
    gitLabConnection('Gitlab')
    timeout(time: 2, unit: 'HOURS')
  }
  environment {
    DOCKER_REGISTRY='registry.petuum.com/internal/scalable-ml/autodist/toolchain'
  }
    agent none
    stages {
        stage('build-image') {
            agent {
                label 'GPU1'
            }
            steps {
                updateGitlabCommitStatus name: 'jenkins', state: 'running'
                sh "docker build --build-arg TF_IMAGE_TAG=1.15.0-gpu-py3 --no-cache -t ${DOCKER_REGISTRY}:tf1 -f docker/Dockerfile.gpu ."
                sh "docker push ${DOCKER_REGISTRY}:tf1"
                sh "docker build --build-arg TF_IMAGE_TAG=2.2.0-gpu --no-cache -t ${DOCKER_REGISTRY}:tf2 -f docker/Dockerfile.gpu ."
                sh "docker push ${DOCKER_REGISTRY}:tf2"
            }
        }
        stage('lint') {
            agent {
                docker { 
                    label 'GPU1'
                    image "${DOCKER_REGISTRY}:tf2"
                }
            }
            steps{
                sh 'prospector autodist'
            }
        }
        stage('test-local') {
            parallel {
                stage('tf1') {
                    agent {
                        docker {
                            alwaysPull true
                            label 'GPU1'
                            image "${DOCKER_REGISTRY}:tf1"
                            args '--gpus all'
                        }
                    }
                    steps{
                        /* .. remove "--run-integration" to have a mini test ..*/
                        sh "cd tests && python3 -m pytest -s --run-integration --junitxml=test_local.xml --cov=autodist --cov-branch --cov-report term-missing --ignore=integration/test_dist.py . && mv .coverage .coverage.local.tf1"
                    }
                    post {
                        success {
                            junit allowEmptyResults: true, testResults: 'tests/test_local.xml'
                            stash includes: 'tests/.coverage.local.tf1', name: 'testcov_local_tf1'
                        }
                    }
                }
                stage('tf2') {
                    agent {
                        docker {
                            alwaysPull true
                            label 'GPU2'
                            image "${DOCKER_REGISTRY}:tf2"
                            args '--gpus all'
                        }
                    }
                    steps{
                        /* .. remove "--run-integration" to have a mini test ..*/
                        sh "cd tests && python3 -m pytest -s --run-integration --junitxml=test_local.xml --cov=autodist --cov-branch --cov-report term-missing --ignore=integration/test_dist.py . && mv .coverage .coverage.local.tf2"
                    }
                    post {
                        success {
                            junit allowEmptyResults: true, testResults: 'tests/test_local.xml'
                            stash includes: 'tests/.coverage.local.tf2', name: 'testcov_local_tf2'
                        }
                    }
                }
            }
        }
        stage('test-distributed') {
            parallel {
                stage('chief') {
                    agent {
                        label 'GPU2'
                    }
                    steps {
                        sh 'docker pull ${DOCKER_REGISTRY}:tf2'
                        sh 'sleep 5'
                        sh 'docker run --gpus all --network=host -v /shared/.ssh:/root/.ssh:ro -v $(pwd)/tests:/mnt -e COVERAGE_PROCESS_START=/mnt/integration/dist.coveragerc ${DOCKER_REGISTRY}:tf2 bash -c "python3 -m pytest -s --junitxml=test_dist.xml integration/test_dist.py"'
                        echo "${myflag}"
                        script {myflag = true}
                        echo "${myflag}"
                    }
                    post {
                        success {
                            junit allowEmptyResults: true, testResults: 'tests/test_dist.xml'
                            stash includes: 'tests/.coverage.*', name: 'testcov_distributed_chief'
                        }
                    }
                }
                stage('worker') {
                    agent {
                        label 'GPU1'
                    }
                    steps {
                        sh 'docker pull ${DOCKER_REGISTRY}:tf2'
                        sh 'docker rm -f worker || true'
                        sh 'docker run --gpus all --name worker -d --privileged --network=host -v /shared/.ssh:/root/.ssh -v $(pwd)/tests:/mnt -e COVERAGE_PROCESS_START=/mnt/integration/dist.coveragerc ${DOCKER_REGISTRY}:tf2 bash -c "env | grep COVERAGE >> /etc/environment && /usr/sbin/sshd -p 12345; sleep infinity"'
                        echo "${myflag}"
                        waitUntil {script {return myflag}}
                        echo "${myflag}"
                    }
                    post {
                        success {
                            stash includes: 'tests/.coverage.*', name: 'testcov_distributed_worker'
                        }
                    }
                }
            }
        }
        stage('report-coverage') {
            agent {
                docker {
                    label 'GPU2'
                    image "${DOCKER_REGISTRY}:tf2"
                }
            }
            steps{
                sh 'mkdir -p coverage-report'
                dir ('coverage-report') {
                    unstash 'testcov_local_tf1'
                    unstash 'testcov_local_tf2'
                    unstash 'testcov_distributed_chief'
                    unstash 'testcov_distributed_worker'
                    sh 'coverage combine tests/'
                    sh 'coverage report'
                    sh 'coverage html -d htmlcov'
                    sh "tar -zcvf htmlcov.tar.gz htmlcov"
                }
            }
             post {
                success {
                    archiveArtifacts allowEmptyArchive: true, artifacts: 'coverage-report/htmlcov.tar.gz', fingerprint: true
                }
            }
        }
    }
    post {
        success {
            updateGitlabCommitStatus name: 'jenkins', state: 'success'
        }
        failure {
            updateGitlabCommitStatus name: 'jenkins', state: 'failed'
        }
        aborted {
            updateGitlabCommitStatus name: 'jenkins', state: 'canceled'
        }
    }
}
