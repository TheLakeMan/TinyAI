/* Tcl in ~ 500 lines of code.
 *
 * Copyright (c) 2007-2016, Salvatore Sanfilippo <antirez at gmail dot com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "picol.h" // Include the header to get typedefs and structs
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Add function pointer type definition for picolCmdFunc
typedef int (*picolCmdFunc)(struct picolInterp *i, int argc, char **argv, void *privdata);

// Forward declarations
int picolCommandCallProc(struct picolInterp *i, int argc, char **argv, void *pd);

// Internal parser token types
enum { PT_ESC, PT_STR, PT_CMD, PT_VAR, PT_SEP, PT_EOL, PT_EOF };

// This struct is internal to picol.c and not defined in picol.h
struct picolParser {
    char *text;
    char *p;           /* current text position */
    int   len;         /* remaining length */
    char *start;       /* token start */
    char *end;         /* token end */
    int   type;        /* token type, PT_... */
    int   insidequote; /* True if inside " " */
};

void picolInitParser(struct picolParser *p, char *text)
{
    p->text = p->p = text;
    p->len         = strlen(text);
    p->start       = 0;
    p->end         = 0;
    p->insidequote = 0;
    p->type        = PT_EOL;
}

// Internal parser functions - keep struct picolParser *
int picolParseSep(struct picolParser *p)
{
    p->start = p->p;
    while (*p->p == ' ' || *p->p == '\t' || *p->p == '\n' || *p->p == '\r') {
        p->p++;
        p->len--;
    }
    p->end  = p->p - 1;
    p->type = PT_SEP;
    return PICOL_OK;
}

int picolParseEol(struct picolParser *p)
{
    p->start = p->p;
    while (*p->p == ' ' || *p->p == '\t' || *p->p == '\n' || *p->p == '\r' || *p->p == ';') {
        p->p++;
        p->len--;
    }
    p->end  = p->p - 1;
    p->type = PT_EOL;
    return PICOL_OK;
}

int picolParseCommand(struct picolParser *p)
{
    int level  = 1;
    int blevel = 0;
    p->start   = ++p->p;
    p->len--;
    while (1) {
        if (p->len == 0) {
            break;
        }
        else if (*p->p == '[' && blevel == 0) {
            level++;
        }
        else if (*p->p == ']' && blevel == 0) {
            if (!--level)
                break;
        }
        else if (*p->p == '\\') {
            p->p++;
            p->len--;
        }
        else if (*p->p == '{') {
            blevel++;
        }
        else if (*p->p == '}') {
            if (blevel != 0)
                blevel--;
        }
        p->p++;
        p->len--;
    }
    p->end  = p->p - 1;
    p->type = PT_CMD;
    if (*p->p == ']') {
        p->p++;
        p->len--;
    }
    return PICOL_OK;
}

int picolParseVar(struct picolParser *p)
{
    p->start = ++p->p;
    p->len--; /* skip the $ */
    while (1) {
        if ((*p->p >= 'a' && *p->p <= 'z') || (*p->p >= 'A' && *p->p <= 'Z') ||
            (*p->p >= '0' && *p->p <= '9') || *p->p == '_') {
            p->p++;
            p->len--;
            continue;
        }
        break;
    }
    if (p->start == p->p) { /* It's just a single char string "$" */
        p->start = p->end = p->p - 1;
        p->type           = PT_STR;
    }
    else {
        p->end  = p->p - 1;
        p->type = PT_VAR;
    }
    return PICOL_OK;
}

int picolParseBrace(struct picolParser *p)
{
    int level = 1;
    p->start  = ++p->p;
    p->len--;
    while (1) {
        if (p->len >= 2 && *p->p == '\\') {
            p->p++;
            p->len--;
        }
        else if (p->len == 0 || *p->p == '}') {
            level--;
            if (level == 0 || p->len == 0) {
                p->end = p->p - 1;
                if (p->len) {
                    p->p++;
                    p->len--; /* Skip final closed brace */
                }
                p->type = PT_STR;
                return PICOL_OK;
            }
        }
        else if (*p->p == '{')
            level++;
        p->p++;
        p->len--;
    }
    return PICOL_OK; /* unreached */
}

int picolParseString(struct picolParser *p)
{
    int newword = (p->type == PT_SEP || p->type == PT_EOL || p->type == PT_STR);
    if (newword && *p->p == '{')
        return picolParseBrace(p);
    else if (newword && *p->p == '"') {
        p->insidequote = 1;
        p->p++;
        p->len--;
    }
    p->start = p->p;
    while (1) {
        if (p->len == 0) {
            p->end  = p->p - 1;
            p->type = PT_ESC;
            return PICOL_OK;
        }
        switch (*p->p) {
        case '\\':
            if (p->len >= 2) {
                p->p++;
                p->len--;
            }
            break;
        case '$':
        case '[':
            p->end  = p->p - 1;
            p->type = PT_ESC;
            return PICOL_OK;
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case ';':
            if (!p->insidequote) {
                p->end  = p->p - 1;
                p->type = PT_ESC;
                return PICOL_OK;
            }
            break;
        case '"':
            if (p->insidequote) {
                p->end  = p->p - 1;
                p->type = PT_ESC;
                p->p++;
                p->len--;
                p->insidequote = 0;
                return PICOL_OK;
            }
            break;
        }
        p->p++;
        p->len--;
    }
    return PICOL_OK; /* unreached */
}

int picolParseComment(struct picolParser *p)
{
    while (p->len && *p->p != '\n') {
        p->p++;
        p->len--;
    }
    return PICOL_OK;
}

int picolGetToken(struct picolParser *p)
{
    while (1) {
        if (!p->len) {
            if (p->type != PT_EOL && p->type != PT_EOF)
                p->type = PT_EOL;
            else
                p->type = PT_EOF;
            return PICOL_OK;
        }
        switch (*p->p) {
        case ' ':
        case '\t':
        case '\r':
            if (p->insidequote)
                return picolParseString(p);
            return picolParseSep(p);
        case '\n':
        case ';':
            if (p->insidequote)
                return picolParseString(p);
            return picolParseEol(p);
        case '[':
            return picolParseCommand(p);
        case '$':
            return picolParseVar(p);
        case '#':
            if (p->type == PT_EOL) {
                picolParseComment(p);
                continue;
            }
            return picolParseString(p);
        default:
            return picolParseString(p);
        }
    }
    return PICOL_OK; /* unreached */
}

// Implementation of picolCreateInterp (required by picol.h)
picolInterp *picolCreateInterp(void)
{
    picolInterp *i = malloc(sizeof(picolInterp));
    if (!i)
        return NULL;

    i->level     = 0;
    i->callframe = malloc(sizeof(picolCallFrame));
    if (!i->callframe) {
        free(i);
        return NULL;
    }

    i->callframe->vars    = NULL;
    i->callframe->parent  = NULL;
    i->callframe->command = NULL; // Initialize command field
    i->commands           = NULL;
    i->resultString       = _strdup("");
    i->result             = PICOL_OK;
    i->arrays             = NULL; // Initialize arrays field
    i->current            = NULL; // Initialize current field

    return i;
}

// Use the existing picolInitInterp as helper for picolCreateInterp
void picolInitInterp(picolInterp *i)
{
    i->level              = 0;
    i->callframe          = malloc(sizeof(picolCallFrame));
    i->callframe->vars    = NULL;
    i->callframe->parent  = NULL;
    i->callframe->command = NULL; // Initialize command field
    i->commands           = NULL;
    i->resultString       = _strdup("");
    i->result             = PICOL_OK;
    i->arrays             = NULL; // Initialize arrays field
    i->current            = NULL; // Initialize current field
}

// Changed return type to void to match header file
void picolSetResult(picolInterp *i, char *s)
{
    free(i->resultString);
    i->resultString = _strdup(s);
}

// Implementation of picolGetResult (required by picol.h)
char *picolGetResult(picolInterp *i) { return i->resultString; }

// Internal implementation of picolGetVar - but with a different name to avoid conflict
// with the declaration in picol.h
static struct picolVar *picolInternalGetVar(picolInterp *i, char *name)
{
    struct picolVar *v = i->callframe->vars;
    while (v) {
        if (strcmp(v->name, name) == 0)
            return v;
        v = v->next;
    }
    return NULL;
}

// This must match the declaration in picol.h
char *picolGetVar(picolInterp *i, char *name)
{
    struct picolVar *v = picolInternalGetVar(i, name);
    return v ? v->val : NULL;
}

// Use typedef picolInterp *
int picolSetVar(picolInterp *i, char *name, char *val)
{
    struct picolVar *v = picolInternalGetVar(i, name);
    if (v) {
        free(v->val);
        v->val = _strdup(val);
        if (!v->val) {
            picolSetResult(i, "Out of memory updating variable");
            i->result = PICOL_ERR;
            return PICOL_ERR;
        }
    }
    else {
        v = malloc(sizeof(picolVar));
        if (!v) {
            picolSetResult(i, "Out of memory setting variable");
            i->result = PICOL_ERR;
            return PICOL_ERR;
        }
        v->name = _strdup(name);
        v->val  = _strdup(val);
        if (!v->name || !v->val) {
            if (v->name)
                free(v->name);
            if (v->val)
                free(v->val);
            free(v);
            picolSetResult(i, "Out of memory setting variable");
            i->result = PICOL_ERR;
            return PICOL_ERR;
        }
        v->next            = i->callframe->vars;
        i->callframe->vars = v;
    }
    return PICOL_OK;
}

// Implementation of array functions required by picol.h
picolArray *picolInternalGetArray(picolInterp *i, char *name)
{
    picolArray *a = i->arrays;
    while (a) {
        if (strcmp(a->name, name) == 0)
            return a;
        a = a->next;
    }
    return NULL;
}

int picolSetArrayVar(picolInterp *i, char *name, char *key, char *val)
{
    picolArray *a = picolInternalGetArray(i, name);

    // Create array if it doesn't exist
    if (!a) {
        a = malloc(sizeof(picolArray));
        if (!a) {
            picolSetResult(i, "Out of memory creating array");
            i->result = PICOL_ERR;
            return PICOL_ERR;
        }
        a->name   = _strdup(name);
        a->vars   = NULL;
        a->next   = i->arrays;
        i->arrays = a;
    }

    // Create key format: "name(key)"
    char fullname[PICOL_MAX_STR];
    snprintf(fullname, PICOL_MAX_STR, "%s(%s)", name, key);

    // Set the variable using existing mechanism
    return picolSetVar(i, fullname, val);
}

char *picolGetArrayVar(picolInterp *i, char *name, char *key)
{
    // Create key format: "name(key)"
    char fullname[PICOL_MAX_STR];
    snprintf(fullname, PICOL_MAX_STR, "%s(%s)", name, key);

    // Get the variable using existing mechanism
    return picolGetVar(i, fullname);
}

struct picolCmd *picolGetCommand(picolInterp *i, char *name)
{
    struct picolCmd *c = i->commands;
    while (c) {
        if (strcmp(c->name, name) == 0)
            return c;
        c = c->next;
    }
    return NULL;
}

int picolRegisterCommand(picolInterp *i, char *name, picolCmdFunc func, void *privdata)
{
    struct picolCmd *c = picolGetCommand(i, name);
    char             errbuf[1024];
    if (c) {
        snprintf(errbuf, 1024, "Command '%s' already defined", name);
        picolSetResult(i, errbuf);
        i->result = PICOL_ERR;
        return PICOL_ERR;
    }

    c = malloc(sizeof(picolCmd));
    if (c)
        c->name = _strdup(name);
    if (c == NULL || c->name == NULL) {
        if (c)
            free(c->name);
        free(c);
        picolSetResult(i, "Out of memory registering command");
        i->result = PICOL_ERR;
        return PICOL_ERR;
    }
    c->func     = func;
    c->type     = PICOL_COMMAND_TYPE_C; // Set command type from enum in picol.h
    c->privdata = privdata;
    c->next     = i->commands;
    i->commands = c;
    return PICOL_OK;
}

// Implementation of picolUnregisterCommand (required by picol.h)
int picolUnregisterCommand(picolInterp *i, char *name)
{
    struct picolCmd *c    = i->commands;
    struct picolCmd *prev = NULL;

    while (c) {
        if (strcmp(c->name, name) == 0) {
            if (prev == NULL) {
                i->commands = c->next;
            }
            else {
                prev->next = c->next;
            }

            // Free command resources
            free(c->name);
            if (c->type == PICOL_COMMAND_TYPE_C && c->func == picolCommandCallProc) {
                char **procdata = c->privdata;
                if (procdata) {
                    free(procdata[0]);
                    free(procdata[1]);
                    free(procdata);
                }
            }
            free(c);
            return PICOL_OK;
        }
        prev = c;
        c    = c->next;
    }

    picolSetResult(i, "No such command");
    i->result = PICOL_ERR;
    return PICOL_ERR;
}

/* EVAL! */
int picolEval(picolInterp *i, char *t)
{
    struct picolParser p;
    int                argc = 0, j;
    char             **argv = NULL;
    char               errbuf[1024];
    int                retcode = PICOL_OK;
    picolSetResult(i, "");
    i->result = PICOL_OK;
    picolInitParser(&p, t);
    while (1) {
        char *t;
        int   tlen;
        int   prevtype = p.type;
        picolGetToken(&p);
        if (p.type == PT_EOF)
            break;
        tlen = p.end - p.start + 1;
        if (tlen < 0)
            tlen = 0;
        t = malloc(tlen + 1);
        memcpy(t, p.start, tlen);
        t[tlen] = '\0';
        if (p.type == PT_VAR) {
            char *v = picolGetVar(i, t);
            if (!v) {
                snprintf(errbuf, 1024, "No such variable '%s'", t);
                free(t);
                picolSetResult(i, errbuf);
                i->result = PICOL_ERR;
                retcode   = PICOL_ERR;
                goto err;
            }
            free(t);
            t = _strdup(v);
        }
        else if (p.type == PT_CMD) {
            retcode = picolEval(i, t);
            free(t);
            if (retcode != PICOL_OK)
                goto err;
            t = _strdup(i->resultString);
        }
        else if (p.type == PT_ESC) {
            /* TODO: escape handling missing! */
        }
        else if (p.type == PT_SEP) {
            prevtype = p.type;
            free(t);
            continue;
        }
        /* We have a complete command + args. Call it! */
        if (p.type == PT_EOL) {
            struct picolCmd *c;
            free(t);
            prevtype = p.type;
            if (argc) {
                if ((c = picolGetCommand(i, argv[0])) == NULL) {
                    snprintf(errbuf, 1024, "No such command '%s'", argv[0]);
                    picolSetResult(i, errbuf);
                    i->result = PICOL_ERR;
                    retcode   = PICOL_ERR;
                    goto err;
                }
                retcode = c->func(i, argc, argv, c->privdata);
                if (retcode != PICOL_OK)
                    goto err;
            }
            /* Prepare for the next command */
            for (j = 0; j < argc; j++)
                free(argv[j]);
            free(argv);
            argv = NULL;
            argc = 0;
            continue;
        }
        /* We have a new token, append to the previous or as new arg? */
        if (prevtype == PT_SEP || prevtype == PT_EOL) {
            argv       = realloc(argv, sizeof(char *) * (argc + 1));
            argv[argc] = t;
            argc++;
        }
        else { /* Interpolation */
            int oldlen = strlen(argv[argc - 1]), tlen = strlen(t);
            argv[argc - 1] = realloc(argv[argc - 1], oldlen + tlen + 1);
            memcpy(argv[argc - 1] + oldlen, t, tlen);
            argv[argc - 1][oldlen + tlen] = '\0';
            free(t);
        }
        prevtype = p.type;
    }
err:
    for (j = 0; j < argc; j++)
        free(argv[j]);
    free(argv);
    return retcode;
}

/* ACTUAL COMMANDS! */
int picolArityErr(picolInterp *i, char *name)
{
    char buf[1024];
    snprintf(buf, 1024, "Wrong number of args for %s", name);
    picolSetResult(i, buf);
    i->result = PICOL_ERR;
    return PICOL_ERR;
}

int picolCommandMath(picolInterp *i, int argc, char **argv, void *pd)
{
    char buf[64];
    int  a, b, c;
    if (argc != 3)
        return picolArityErr(i, argv[0]);
    a = atoi(argv[1]);
    b = atoi(argv[2]);
    if (argv[0][0] == '+')
        c = a + b;
    else if (argv[0][0] == '-')
        c = a - b;
    else if (argv[0][0] == '*')
        c = a * b;
    else if (argv[0][0] == '/') {
        if (b == 0) {
            picolSetResult(i, "Division by zero");
            i->result = PICOL_ERR;
            return PICOL_ERR;
        }
        c = a / b;
    }
    else if (argv[0][0] == '>' && argv[0][1] == '\0')
        c = a > b;
    else if (argv[0][0] == '>' && argv[0][1] == '=')
        c = a >= b;
    else if (argv[0][0] == '<' && argv[0][1] == '\0')
        c = a < b;
    else if (argv[0][0] == '<' && argv[0][1] == '=')
        c = a <= b;
    else if (argv[0][0] == '=' && argv[0][1] == '=')
        c = a == b;
    else if (argv[0][0] == '!' && argv[0][1] == '=')
        c = a != b;
    else
        c = 0; /* I hate warnings */
    snprintf(buf, 64, "%d", c);
    picolSetResult(i, buf);
    return PICOL_OK;
}

int picolCommandSet(picolInterp *i, int argc, char **argv, void *pd)
{
    if (argc != 3)
        return picolArityErr(i, argv[0]);
    if (picolSetVar(i, argv[1], argv[2]) != PICOL_OK)
        return PICOL_ERR;
    picolSetResult(i, argv[2]);
    return PICOL_OK;
}

int picolCommandPuts(picolInterp *i, int argc, char **argv, void *pd)
{
    if (argc != 2)
        return picolArityErr(i, argv[0]);
    printf("%s\n", argv[1]);
    picolSetResult(i, "");
    return PICOL_OK;
}

int picolCommandIf(picolInterp *i, int argc, char **argv, void *pd)
{
    int retcode;
    if (argc != 3 && argc != 5)
        return picolArityErr(i, argv[0]);
    if ((retcode = picolEval(i, argv[1])) != PICOL_OK)
        return retcode;
    if (atoi(i->resultString))
        return picolEval(i, argv[2]);
    else if (argc == 5)
        return picolEval(i, argv[4]);
    return PICOL_OK;
}

int picolCommandWhile(picolInterp *i, int argc, char **argv, void *pd)
{
    if (argc != 3)
        return picolArityErr(i, argv[0]);
    while (1) {
        int retcode = picolEval(i, argv[1]);
        if (retcode != PICOL_OK)
            return retcode;
        if (atoi(i->resultString)) {
            if ((retcode = picolEval(i, argv[2])) == PICOL_CONTINUE)
                continue;
            else if (retcode == PICOL_OK)
                continue;
            else if (retcode == PICOL_BREAK)
                return PICOL_OK;
            else
                return retcode;
        }
        else {
            return PICOL_OK;
        }
    }
}

int picolCommandRetCodes(picolInterp *i, int argc, char **argv, void *pd)
{
    if (argc != 1)
        return picolArityErr(i, argv[0]);
    picolSetResult(i, "");
    if (strcmp(argv[0], "break") == 0)
        return PICOL_BREAK;
    else if (strcmp(argv[0], "continue") == 0)
        return PICOL_CONTINUE;
    return PICOL_OK;
}

void picolDropCallFrame(picolInterp *i)
{
    picolCallFrame  *cf = i->callframe;
    struct picolVar *v  = cf->vars, *t;
    while (v) {
        t = v->next;
        free(v->name);
        free(v->val);
        free(v);
        v = t;
    }
    i->callframe = cf->parent;
    free(cf);
}

int picolCommandCallProc(picolInterp *i, int argc, char **argv, void *pd)
{
    char          **x = pd, *alist = x[0], *body = x[1], *p = _strdup(alist), *tofree;
    picolCallFrame *cf    = malloc(sizeof(picolCallFrame));
    int             arity = 0, done = 0, errcode = PICOL_OK;
    char            errbuf[1024];
    cf->vars     = NULL;
    cf->parent   = i->callframe;
    cf->command  = _strdup(argv[0]); // Set the currently executing command
    i->callframe = cf;
    tofree       = p;
    while (1) {
        char *start = p;
        while (*p != ' ' && *p != '\0')
            p++;
        if (*p != '\0' && p == start) {
            p++;
            continue;
        }
        if (p == start)
            break;
        if (*p == '\0')
            done = 1;
        else
            *p = '\0';
        if (++arity > argc - 1)
            goto arityerr;
        picolSetVar(i, start, argv[arity]);
        p++;
        if (done)
            break;
    }
    free(tofree);
    if (arity != argc - 1)
        goto arityerr;
    errcode = picolEval(i, body);
    if (errcode == PICOL_RETURN)
        errcode = PICOL_OK;
    picolDropCallFrame(i); /* remove the called proc callframe */
    return errcode;
arityerr:
    snprintf(errbuf, 1024, "Proc '%s' called with wrong arg num", argv[0]);
    picolSetResult(i, errbuf);
    i->result = PICOL_ERR;
    free(tofree);
    picolDropCallFrame(i); /* remove the called proc callframe */
    return PICOL_ERR;
}

int picolCommandProc(picolInterp *i, int argc, char **argv, void *pd)
{
    char **procdata = malloc(sizeof(char *) * 2);
    if (!procdata) {
        picolSetResult(i, "Out of memory registering proc");
        i->result = PICOL_ERR;
        return PICOL_ERR;
    }
    procdata[0] = procdata[1] = NULL;

    if (argc != 4) {
        free(procdata);
        return picolArityErr(i, argv[0]);
    }

    procdata[0] = _strdup(argv[2]); /* Arguments list. */
    procdata[1] = _strdup(argv[3]); /* Procedure body. */
    if (procdata[0] == NULL || procdata[1] == NULL)
        goto oom;
    int regResult = picolRegisterCommand(i, argv[1], picolCommandCallProc, procdata);
    if (regResult == PICOL_OK) {
        picolSetResult(i, "");
    }
    return regResult;

oom:
    free(procdata[0]);
    free(procdata[1]);
    free(procdata);
    picolSetResult(i, "Out of memory registering proc");
    i->result = PICOL_ERR;
    return PICOL_ERR;
}

int picolCommandReturn(picolInterp *i, int argc, char **argv, void *pd)
{
    if (argc != 1 && argc != 2)
        return picolArityErr(i, argv[0]);
    picolSetResult(i, (argc == 2) ? argv[1] : "");
    return PICOL_RETURN;
}

/* Register core commands */
void picolRegisterCoreCommands(picolInterp *i)
{
    int   j;
    char *name[] = {"+", "-", "*", "/", ">", ">=", "<", "<=", "==", "!="};
    for (j = 0; j < (int)(sizeof(name) / sizeof(char *)); j++)
        picolRegisterCommand(i, name[j], picolCommandMath, NULL);
    picolRegisterCommand(i, "set", picolCommandSet, NULL);
    picolRegisterCommand(i, "puts", picolCommandPuts, NULL);
    picolRegisterCommand(i, "if", picolCommandIf, NULL);
    picolRegisterCommand(i, "while", picolCommandWhile, NULL);
    picolRegisterCommand(i, "break", picolCommandRetCodes, NULL);
    picolRegisterCommand(i, "continue", picolCommandRetCodes, NULL);
    picolRegisterCommand(i, "proc", picolCommandProc, NULL);
    picolRegisterCommand(i, "return", picolCommandReturn, NULL);
}

/* Add an implementation for picolFreeInterp */
void picolFreeInterp(picolInterp *i)
{
    if (!i)
        return;

    // Free all variables in all call frames
    while (i->callframe) {
        picolDropCallFrame(i);
    }

    // Free commands
    struct picolCmd *c = i->commands;
    struct picolCmd *next;
    while (c) {
        next = c->next;
        free(c->name);
        if (c->type == PICOL_COMMAND_TYPE_C && c->func == picolCommandCallProc) {
            char **procdata = c->privdata;
            if (procdata) {
                free(procdata[0]);
                free(procdata[1]);
                free(procdata);
            }
        }
        free(c);
        c = next;
    }

    // Free arrays
    picolArray *a = i->arrays;
    picolArray *nextArray;
    while (a) {
        nextArray = a->next;
        free(a->name);
        // Variables inside arrays are already freed by picolDropCallFrame
        free(a);
        a = nextArray;
    }

    // Free result string
    free(i->resultString);

    // Free current command pointer if set
    if (i->current) {
        free(i->current);
    }
}
