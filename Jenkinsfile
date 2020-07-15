def myflag = false

pipeline {
    options {
        timeout(time: 2, unit: 'HOURS')
    }

    environment {
        DOCKER_REGISTRY='registry.petuum.com/internal/scalable-ml/autodist/toolchain'
    }

    agent { label 'petuum-jenkins-slave' }

    stages {
        stage('begin') {
            steps {
                setBuildStatus("Build in progress", "pending", "${env.GIT_COMMIT}", "${env.BUILD_URL}");
            }
        }
        stage('build-image') {
            agent {
                label 'GPU1'
            }
            steps {
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
                        always {
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
                        always {
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
                        always {
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
                        always {
                            sh 'docker rm -f worker || true'
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
            setBuildStatus("Build succeeded", "success", "${env.GIT_COMMIT}", "${env.BUILD_URL}");
        }
        failure {
            setBuildStatus("Build failed", "failure", "${env.GIT_COMMIT}", "${env.BUILD_URL}");
        }
    }
}

/*
 * We've disabled the github commit notification from plugin because it doesn't
 * allow us to overwrite the target URL (it takes raw hostname of jenkins server,
 * which we dont want, as we're pointing to public hostname instead).
 *
 * In future, we can use setBuildStatusViaPlugin() instead, but until those plugins
 * become more stable, can use this.
 */
void setBuildStatus(String message, String state, String gitCommit, String buildUrl) {
    withCredentials([string(credentialsId: 'petuumops-github-service-token', variable: 'TOKEN')]) {
        String statusUrl = "https://api.github.com/repos/petuum/autodist/statuses/$gitCommit"
        String targetUrl = buildUrl.replaceAll(/http.*\.(com|io)\//,"https://jenkins.petuum.io/")

        // Can enable 'set -x' for debugging. TOKEN is not logged
        sh """
            # set -x
            curl -H \"Authorization: token $TOKEN\" \
                 -X POST \
                 -d '{\"description\": \"$message\", \
                      \"state\": \"$state\", \
                      \"context\": "ci/jenkins", \
                      \"target_url\" : \"$targetUrl\" }' \
                 ${statusUrl}
        """
    }
}

/*
 * Don't use this until plugins become more stable, for example:
 * https://issues.jenkins-ci.org/browse/JENKINS-54249.
 *
 * Instead, opt for setBuildStatus() and disable from job (we need to send
 * our own notification because auto won't allow us to set the public revere proxy URL)
 */
void setBuildStatusViaPlugin(String message, String state, String gitCommit, String repoSource, String buildUrl) {
    String targetUrl = buildUrl.replaceAll(/http.*\.(com|io)\//,"https://jenkins.petuum.io/")
    step([
        $class: "GitHubCommitStatusSetter",
        contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "ci/jenkins/branch"],
        reposSource: [$class: "ManuallyEnteredRepositorySource", url: repoSource],
        errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "error"]],
        statusBackrefSource: [$class: "ManuallyEnteredBackrefSource", backref: targetUrl],
        statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
    ]);
}
