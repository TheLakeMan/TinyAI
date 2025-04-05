/**
 * TinyAI Picol Header
 * 
 * This header defines the core data structures and functions for the Picol
 * interpreter that serves as the foundation of TinyAI.
 */

#ifndef TINYAI_PICOL_H
#define TINYAI_PICOL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------- Base interpreter structure ----------------------- */

/* PICOL_MAX_STR is the maximum length of a string contained in a picolVar or
 * processed as a command result or argument. */
#define PICOL_MAX_STR 4096

enum {PICOL_OK, PICOL_ERR, PICOL_RETURN, PICOL_BREAK, PICOL_CONTINUE};
enum {PICOL_COMMAND_TYPE_STR, PICOL_COMMAND_TYPE_C};

typedef struct picolVar {
    char *name, *val;
    struct picolVar *next;
} picolVar;

typedef struct picolArray {
    char *name;
    struct picolVar *vars;
    struct picolArray *next;
} picolArray;

typedef struct picolCallFrame {
    struct picolVar *vars;
    struct picolCallFrame *parent; /* parent is NULL at top level */
    char *command;                 /* Currently executing command (or NULL) */
} picolCallFrame;

typedef struct picolInterp {
    int result;
    char *resultString;
    struct picolCmd *commands;
    struct picolCallFrame *callframe;
    struct picolArray *arrays;
    int level;
    char *current;
} picolInterp;

typedef struct picolCmd {
    char *name;
    int (*func)(struct picolInterp *i, int argc, char **argv, void *privdata);
    void *privdata;
    struct picolCmd *next;
    int type;
} picolCmd;

/* ----------------------- Function Prototypes ----------------------- */

/* Parser and evaluator */
int picolParse(picolInterp *i, char *t, char **start, char *end);
int picolEval(picolInterp *i, char *t);

/* Variables and arrays */
int picolSetVar(picolInterp *i, char *name, char *val);
char *picolGetVar(picolInterp *i, char *name);
int picolSetArrayVar(picolInterp *i, char *name, char *key, char *val);
char *picolGetArrayVar(picolInterp *i, char *name, char *key);

/* Commands */
int picolRegisterCommand(picolInterp *i, char *name, int (*func)(picolInterp*,int,char**,void*), void *privdata);
int picolUnregisterCommand(picolInterp *i, char *name);
int picolCommandMath(picolInterp *i, int argc, char **argv, void *pd);
int picolCommandSet(picolInterp *i, int argc, char **argv, void *pd);
int picolCommandPuts(picolInterp *i, int argc, char **argv, void *pd);
int picolCommandIf(picolInterp *i, int argc, char **argv, void *pd);
int picolCommandWhile(picolInterp *i, int argc, char **argv, void *pd);
int picolCommandRetCodes(picolInterp *i, int argc, char **argv, void *pd);

/* Results */
void picolSetResult(picolInterp *i, char *s); // Changed return type to void
char *picolGetResult(picolInterp *i);

/* Initialization and cleanup */
picolInterp *picolCreateInterp(void);
void picolFreeInterp(picolInterp *i);
void picolRegisterCoreCommands(picolInterp *i);

#ifdef __cplusplus
}
#endif

#endif /* TINYAI_PICOL_H */
