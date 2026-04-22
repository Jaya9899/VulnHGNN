/*
 * Deadlock Detection & Recovery (simplified)
 * Compile: gcc -o detect deadlock_simple.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#define N 4          /* processes */
#define M 3          /* resource types */
#define ITER 4       /* requests per process */

int Available[M], Alloc[N][M], Req[N][M];
int terminated[N], blocked[N], reply[N], all_done;

pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  mgr_cv = PTHREAD_COND_INITIALIZER;
pthread_cond_t  proc_cv[N];

/* Simple request queue */
int qpid[16], qreq[16][M], qhead, qtail;

/* ── Manager: grant or block each resource request ── */
void *manager(void *_) {
    pthread_mutex_lock(&mtx);
    while (1) {
        while (qhead == qtail && !all_done)
            pthread_cond_wait(&mgr_cv, &mtx);
        if (all_done && qhead == qtail) break;

        int pid = qpid[qhead % 16];
        int *req = qreq[qhead++ % 16];

        if (terminated[pid]) { reply[pid] = 0; pthread_cond_signal(&proc_cv[pid]); continue; }

        /* Check if request fits available resources */
        int ok = 1;
        for (int j = 0; j < M; j++) if (req[j] > Available[j]) { ok = 0; break; }

        if (ok) {
            for (int j = 0; j < M; j++) {
                Available[j] -= req[j];
                Alloc[pid][j] += req[j];
                Req[pid][j] = 0;
            }
            blocked[pid] = 0; reply[pid] = 1;
            printf("[Manager] P%d GRANTED\n", pid);
            pthread_cond_signal(&proc_cv[pid]);
        } else {
            memcpy(Req[pid], req, M * sizeof(int));
            blocked[pid] = 1;
            printf("[Manager] P%d BLOCKED\n", pid);
        }
    }
    pthread_mutex_unlock(&mtx);
    return NULL;
}

/* ── Detector: find and break deadlocks every 0.5s ── */
void *detector(void *_) {
    while (!all_done) {
        usleep(500000);
        pthread_mutex_lock(&mtx);

        /* Detection: simulate resource release order (Banker-style) */
        int Work[M], Finish[N];
        memcpy(Work, Available, sizeof(Work));
        for (int i = 0; i < N; i++) {
            Finish[i] = terminated[i];
            /* Processes holding nothing can always finish */
            if (!Finish[i]) {
                int holds = 0;
                for (int j = 0; j < M; j++) if (Alloc[i][j]) { holds = 1; break; }
                if (!holds) Finish[i] = 1;
            }
        }
        /* Repeatedly find a process whose request can be satisfied */
        for (int found = 1; found; ) {
            found = 0;
            for (int i = 0; i < N; i++) {
                if (Finish[i]) continue;
                int ok = 1;
                for (int j = 0; j < M; j++) if (Req[i][j] > Work[j]) { ok = 0; break; }
                if (ok) {
                    for (int j = 0; j < M; j++) Work[j] += Alloc[i][j];
                    Finish[i] = 1; found = 1;
                }
            }
        }

        /* Collect deadlocked processes */
        int dl[N], count = 0;
        for (int i = 0; i < N; i++) dl[i] = !Finish[i];
        for (int i = 0; i < N; i++) if (dl[i]) count++;

        if (!count) { printf("[Detector] No deadlock\n"); pthread_mutex_unlock(&mtx); continue; }

        printf("[Detector] DEADLOCK: ");
        for (int i = 0; i < N; i++) if (dl[i]) printf("P%d ", i);
        printf("\n");

        /* Recovery: kill cheapest victim (smallest allocation) until resolved */
        while (count--) {
            int victim = -1, min = 999;
            for (int i = 0; i < N; i++) {
                if (!dl[i]) continue;
                int s = 0; for (int j = 0; j < M; j++) s += Alloc[i][j];
                if (s < min) { min = s; victim = i; }
            }
            printf("[Detector] Terminating P%d\n", victim);
            for (int j = 0; j < M; j++) { Available[j] += Alloc[victim][j]; Alloc[victim][j] = Req[victim][j] = 0; }
            terminated[victim] = blocked[victim] = 0;
            reply[victim] = 0;
            pthread_cond_signal(&proc_cv[victim]);
            dl[victim] = 0;
        }

        /* Unblock any process that can now proceed */
        for (int i = 0; i < N; i++) {
            if (!blocked[i] || terminated[i]) continue;
            int ok = 1;
            for (int j = 0; j < M; j++) if (Req[i][j] > Available[j]) { ok = 0; break; }
            if (ok) {
                for (int j = 0; j < M; j++) { Available[j] -= Req[i][j]; Alloc[i][j] += Req[i][j]; Req[i][j] = 0; }
                blocked[i] = 0; reply[i] = 1;
                printf("[Detector] Unblocked P%d\n", i);
                pthread_cond_signal(&proc_cv[i]);
            }
        }
        pthread_mutex_unlock(&mtx);
    }
    return NULL;
}

/* ── Process: repeatedly request and release resources ── */
void *process(void *arg) {
    int pid = *(int *)arg; free(arg);

    for (int iter = 0; iter < ITER; iter++) {
        usleep(50000 + rand() % 200000);
        pthread_mutex_lock(&mtx);
        if (terminated[pid]) { pthread_mutex_unlock(&mtx); goto done; }

        /* Build and enqueue request */
        int *req = qreq[qtail % 16];
        for (int j = 0; j < M; j++) req[j] = rand() % 4;
        qpid[qtail++ % 16] = pid;
        printf("[P%d] requesting:", pid);
        for (int j = 0; j < M; j++) printf(" %d", req[j]);
        printf("\n");

        reply[pid] = -1;
        pthread_cond_signal(&mgr_cv);
        while (reply[pid] == -1) pthread_cond_wait(&proc_cv[pid], &mtx);
        if (reply[pid] == 0) { pthread_mutex_unlock(&mtx); printf("[P%d] terminated\n", pid); goto done; }
        pthread_mutex_unlock(&mtx);

        usleep(100000);   /* use resources */

        pthread_mutex_lock(&mtx);
        if (!terminated[pid]) {
            for (int j = 0; j < M; j++) { Available[j] += Alloc[pid][j]; Alloc[pid][j] = 0; }
            printf("[P%d] released\n", pid);
        }
        pthread_mutex_unlock(&mtx);
    }

done:
    pthread_mutex_lock(&mtx);
    terminated[pid] = 1;
    int everyone = 1;
    for (int i = 0; i < N; i++) if (!terminated[i]) { everyone = 0; break; }
    if (everyone) { all_done = 1; pthread_cond_signal(&mgr_cv); }
    pthread_mutex_unlock(&mtx);
    printf("[P%d] done\n", pid);
    return NULL;
}

int main() {
    srand(time(NULL));
    for (int i = 0; i < N; i++) pthread_cond_init(&proc_cv[i], NULL);
    for (int j = 0; j < M; j++) Available[j] = 2 + rand() % 4;  /* low → triggers deadlocks */
    memset(Alloc, 0, sizeof(Alloc)); memset(Req, 0, sizeof(Req));
    memset(terminated, 0, sizeof(terminated)); memset(blocked, 0, sizeof(blocked));

    printf("Available:"); for (int j = 0; j < M; j++) printf(" %d", Available[j]); printf("\n\n");

    pthread_t mgr, det, ptid[N];
    pthread_create(&mgr, NULL, manager, NULL);
    pthread_create(&det, NULL, detector, NULL);
    for (int i = 0; i < N; i++) {
        int *id = malloc(sizeof(int)); *id = i;
        pthread_create(&ptid[i], NULL, process, id);
    }
    for (int i = 0; i < N; i++) pthread_join(ptid[i], NULL);
    pthread_mutex_lock(&mtx);
    all_done = 1; pthread_cond_signal(&mgr_cv);
    pthread_mutex_unlock(&mtx);
    pthread_join(mgr, NULL);
    pthread_join(det, NULL);
    printf("\n=== Done ===\n");
    return 0;
}