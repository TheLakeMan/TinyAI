#ifndef TINYAI_WEB_SERVER_H
#define TINYAI_WEB_SERVER_H

#include "../core/picol.h" // Include picol for interpreter access if needed

/**
 * @brief Starts the web server.
 *
 * Initializes and runs the Mongoose web server loop.
 *
 * @param interp The picol interpreter instance (can be NULL if not needed).
 * @param port The port number to listen on (e.g., "8080").
 * @param document_root The path to the directory containing web UI files.
 * @return 0 on successful shutdown, non-zero on error.
 */
int start_web_server(picolInterp *interp, const char *port, const char *document_root);

#endif // TINYAI_WEB_SERVER_H
