/*
 * This file is part of xrayutilities.
 *
 * xrayutilities is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * Copyright (C) 2020 Dominik Kriegner <dominik.kriegner@gmail.com>
*/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "xrayutilities.h"

typedef char *multi_tok_t;
typedef int (*fp_check)(long);

char *multi_tok(char *haystack, multi_tok_t *string, char *needle) {
    /* multi character tokenizer similar to strtok
     *
     * an internal reference to the search string is saved and the input string
     * is modified!
     *
     * see: https://stackoverflow.com/a/29789623
     */
    if (haystack != NULL)
        *string = haystack;

    if (*string == NULL)
        return *string;

    char *end = strstr(*string, needle);
    if (end == NULL) {
        char *temp = *string;
        *string = NULL;
        return temp;
    }

    char *temp = *string;

    *end = '\0';
    *string = end + strlen(needle);
    return temp;
}

multi_tok_t mtinit() { return NULL; }

int check2n(long h) {
    if(h % 2 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check2np1(long h) {
    if((h-1) % 2 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check3n(long h) {
    if(h % 3 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check3np1(long h) {
    if((h-1) % 3 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check3np2(long h) {
    if((h-2) % 3 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check4n(long h) {
    if(h % 4 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check4np2(long h) {
    if((h-2) % 4 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check6n(long h) {
    if(h % 6 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8n(long h) {
    if(h % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8np1(long h) {
    if((h-1) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8nm1(long h) {
    if((h+1) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8np3(long h) {
    if((h-3) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8nm3(long h) {
    if((h+3) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8np4(long h) {
    if((h-4) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8np5(long h) {
    if((h-5) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int check8np7(long h) {
    if((h-7) % 8 == 0) {
        return 1;
    } else {
        return 0;
    }
}

int hklpattern_applies(long *hkl, const char *condhkl) {
    /*
     * helper function to determine if Miller indices fit a certain pattern
     *
     * Parameters
     * ----------
     *  hkl : array of three integers Miller indices
     *  condhkl : condition string similar to 'hkl', 'hh0', or '0k0'
     *
     * Returns
     * -------
     *  1 if hkl fulfills the pattern, 0 otherwise
    */
    int n=0;

    if(condhkl[n] == '0' && hkl[0] != 0) {
        return 0;
    }
    n = n + 1;
    if(condhkl[n] == '-') {
        n = n + 1;
        if(condhkl[n] == 'h' && hkl[1] != -hkl[0]) {
            return 0;
        }
    }
    else if(condhkl[n] == '0' && hkl[1] != 0) {
        return 0;
    }
    else if(condhkl[n] == 'h' && hkl[1] != hkl[0]) {
        return 0;
    }
    /*n = n + 1;
    if(condhkl[n] == '(') {
        do {
            n = n + 1;
        } while(condhkl[n] != ')');
        n = n + 1;
    }
    else if(condhkl[n] == 'i') {
        n = n + 1;
    }*/
    if(condhkl[strlen(condhkl)-1] == '0' && hkl[2] != 0) {
        return 0;
    }
    return 1;
}

int reflection_condition_met(long *hkl, const char *cond) {
    /*
     * helper function to determine allowed Miller indices
     *
     * Parameters
     * ----------
     *  hkl: list or tuple
     *   Miller indices of the reflection
     *  cond: str
     *   condition string similar to 'h+k=2n, h+l,k+l=2n'
     *
     * Returns
     * -------
     *  1 if condition is met, 0 otherwise
    */
    int fulfilled = 0;
    const char equal[2] = "=", comma[2] = ",";
    char or[5] = " or ";
    char commaspace[3] = ", ";
    char *tsubcond, *texpr, *lexpr, *rexpr, *l;
    multi_tok_t ssubcond=mtinit(), sexpr=mtinit();
    fp_check checkfunc;
    /* string buffer to avoid changing the argument */
    char *input = malloc((strlen(cond)+1) * sizeof(char));

    strcpy(input, cond);

    /* split at ' or '
     * at least one subcond needs to be fulfilled to return 1
     * */
    tsubcond = multi_tok(input, &ssubcond, or);
    while(tsubcond != NULL) {
        //printf("%s\n", tsubcond);
        fulfilled = 1;
        /* split various subconditions at ', '
         * all condititions need to be fulfilled
         * */
        texpr = multi_tok(tsubcond, &sexpr, commaspace);
        while(texpr != NULL) {
            //printf("\t%s\n", texpr);
            lexpr = strtok(texpr, equal);
            rexpr = strtok(NULL, equal);
            //printf("\t\t%s %s\n", lexpr, rexpr);
            if(strcmp(rexpr, "2n") == 0) {
                checkfunc = &check2n;
            }
            else if(strcmp(rexpr, "2n+1") == 0) {
                checkfunc = &check2np1;
            }
            else if(strcmp(rexpr, "3n") == 0) {
                checkfunc = &check3n;
            }
            else if(strcmp(rexpr, "3n+1") == 0) {
                checkfunc = &check3np1;
            }
            else if(strcmp(rexpr, "3n+2") == 0) {
                checkfunc = &check3np2;
            }
            else if(strcmp(rexpr, "4n") == 0) {
                checkfunc = &check4n;
            }
            else if(strcmp(rexpr, "4n+2") == 0) {
                checkfunc = &check4np2;
            }
            else if(strcmp(rexpr, "6n") == 0) {
                checkfunc = &check6n;
            }
            else if(strcmp(rexpr, "8n") == 0) {
                checkfunc = &check8n;
            }
            else if(strcmp(rexpr, "8n+1") == 0) {
                checkfunc = &check8np1;
            }
            else if(strcmp(rexpr, "8n-1") == 0) {
                checkfunc = &check8nm1;
            }
            else if(strcmp(rexpr, "8n+3") == 0) {
                checkfunc = &check8np3;
            }
            else if(strcmp(rexpr, "8n-3") == 0) {
                checkfunc = &check8nm3;
            }
            else if(strcmp(rexpr, "8n+4") == 0) {
                checkfunc = &check8np4;
            }
            else if(strcmp(rexpr, "8n+5") == 0) {
                checkfunc = &check8np5;
            }
            else if(strcmp(rexpr, "8n+7") == 0) {
                checkfunc = &check8np7;
            }
            else {
		char errorstring[100];
		sprintf(errorstring, "Right hand side of reflection condition (%s) not implemented", rexpr);
                PyErr_SetString(PyExc_RuntimeError, errorstring);
		free(input);
		return -1;
            }
            /* split left expression at ',' */
            l = strtok(lexpr, comma);
            while(l != NULL) {
                //printf("\t\t\t%s\n", l);
                if(strcmp(l, "h") == 0) {
                    if(checkfunc(hkl[0]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "k") == 0) {
                    if(checkfunc(hkl[1]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "l") == 0) {
                    if(checkfunc(hkl[2]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "h+k") == 0) {
                    if(checkfunc(hkl[0] + hkl[1]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "h-k") == 0) {
                    if(checkfunc(hkl[0] - hkl[1]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "-h+k") == 0) {
                    if(checkfunc(-hkl[0] + hkl[1]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "h+l") == 0) {
                    if(checkfunc(hkl[0] + hkl[2]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "k+l") == 0) {
                    if(checkfunc(hkl[1] + hkl[2]) == 0) { fulfilled = 0; }
                }
                else if(strcmp(l, "h+k+l") == 0) {
                    if(checkfunc(hkl[0] + hkl[1] + hkl[2]) == 0) {
                        fulfilled = 0;
                    }
                }
                else if(strcmp(l, "-h+k+l") == 0) {
                    if(checkfunc(-hkl[0] + hkl[1] + hkl[2]) == 0) {
                        fulfilled = 0;
                    }
                }
                else if(strcmp(l, "2h+l") == 0) {
                    if(checkfunc(2*hkl[0] + hkl[2]) == 0) {
                        fulfilled = 0;
                    }
                }
                else if(strcmp(l, "2k+l") == 0) {
                    if(checkfunc(2*hkl[1] + hkl[2]) == 0) {
                        fulfilled = 0;
                    }
                }
                //printf("\t\t\t%d\n", fulfilled);
                if(fulfilled == 0) { break; }
                l = strtok(NULL, comma);
            }
            if(fulfilled == 0) { break; }
            texpr = multi_tok(NULL, &sexpr, commaspace);
        }
        if(fulfilled == 1) {
	    free(input);
            return 1;
        }
        tsubcond = multi_tok(NULL, &ssubcond, or);
    }
    free(input);
    return 0;
}

PyObject* testhklcond(PyObject *self, PyObject *args) {
    /*
     * test if a Bragg peak is allowed according to reflection conditions
     *
     * Parameters
     * ----------
     *  hkl :           Miller indices of the peak to test (integer array)
     *  condgeneral :   General reflection conditions (list of tuples)
     *  condwp :        Reflection conditions for Wyckoff positions
     *                  (list of list of tuples)
     *
     * Returns
     * -------
     * bool : True if peak is allowed, False otherwise
     */
    int i, j, level, r;
    int pattern_applied = 0, condition_met = 0;
    int pattern_appliedwp = 0, condition_metwp = 0;
    long hkl[3];
    const char *hklpattern, *cond;
    PyObject *hkls, *pyhkl, *condgeneral, *condwp, *subcond, *e;
    PyObject *iterhkl;
    Py_ssize_t n, m, dummy;

    /* Python argument conversion code */
    if (!PyArg_ParseTuple(args, "O!O!O!",
                          &PySet_Type, &hkls,
                          &PyList_Type, &condgeneral,
                          &PyList_Type, &condwp)) {
        return NULL;
    }

    /* test general reflection conditions
     * if they are violated the peak is forbidden
     */
    n = PyList_Size(condgeneral);
    iterhkl = PyObject_GetIter(hkls);
    while ((pyhkl = PyIter_Next(iterhkl))) {
        for (i = 0; i < 3; ++i) {
            hkl[i] = PyLong_AsLong(PyTuple_GetItem(pyhkl, i));
        }
        Py_DECREF(pyhkl);
        //printf("(%ld %ld %ld)\n", hkl[0], hkl[1], hkl[2]);
        /* iterate over condgeneral */
        for (i = 0; i < n; ++i) {
            subcond = PyList_GET_ITEM(condgeneral, i);
            hklpattern = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(subcond, 0), &dummy);
            cond = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(subcond, 1), &dummy);
            if (hklpattern_applies(hkl, hklpattern) == 1) {
                pattern_applied = 1;
		r = reflection_condition_met(hkl, cond);
                if (r == 1) {
                    condition_met = 1;
                } else {
                    Py_DECREF(iterhkl);
	            if (r == 0) {Py_RETURN_FALSE;}
                    else {return (PyObject *) NULL;}
                }
            }
        }
    }
    Py_DECREF(iterhkl);

    /* if there are no special conditions for at least one Wyckoff position
     * then directly return
     */
    n = PyList_Size(condwp);
    for (i = 0; i < n; ++i) {
        subcond = PyList_GET_ITEM(condwp, i);
        if (subcond == Py_None) {
            if (condition_met == 1) {
                Py_RETURN_TRUE;
            } else {
                if (pattern_applied == 1) {
                    Py_RETURN_FALSE;
                } else {
                    Py_RETURN_TRUE;
                }
            }
        }
    }
    /* test specific conditions for the Wyckoff positions */
    n = PyList_Size(condwp);
    for (j = 0; j < n; ++j) {
        e = PyList_GET_ITEM(condwp, j);
        m = PyList_Size(e);
        level = 100;
        iterhkl = PyObject_GetIter(hkls);
        while ((pyhkl = PyIter_Next(iterhkl))) {
            for (i = 0; i < 3; ++i) {
                hkl[i] = PyLong_AsLong(PyTuple_GetItem(pyhkl, i));
            }
            //printf("(%ld %ld %ld)\n", hkl[0], hkl[1], hkl[2]);
            Py_DECREF(pyhkl);
            for (i = 0; i < m; ++i) {
                subcond = PyList_GET_ITEM(e, m-1-i);
                hklpattern = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(subcond, 0), &dummy);
                cond = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(subcond, 1), &dummy);
                //printf("%s %s\n", hklpattern, cond);
                if (hklpattern_applies(hkl, hklpattern) == 1) {
                    pattern_appliedwp = 1;
                    r = reflection_condition_met(hkl, cond);
                    if (r == 1) {
                        if (i <= level) {
                            condition_metwp = 1;
                            level = i;
                        }
                    } else {
                        if (r < 0) {
                            Py_DECREF(iterhkl);
                            return (PyObject *) NULL;
                        }
                        if (i < level) {
                            condition_metwp = 0;
                            level = i;
                            break;
                        }
                    }
                }
            }
        }
        Py_DECREF(iterhkl);
        if (condition_metwp == 1) {
            break;
        }
    }

    if (pattern_appliedwp == 1) {
        if (condition_metwp == 1) {
            Py_RETURN_TRUE;
        } else {
            Py_RETURN_FALSE;
        }
    } else if (condition_met == 1 || pattern_applied == 0) {
        Py_RETURN_TRUE;
    } else {
        if (pattern_applied == 1) {
            Py_RETURN_FALSE;
        } else {
            Py_RETURN_TRUE;
        }
    }
}
