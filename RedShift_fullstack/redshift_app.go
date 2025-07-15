package main

import (
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

func main() {
	// Load environment variables
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// Set up HTTP handler
	http.HandleFunc("/", handleColumns)

	fmt.Println("Server starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleColumns(w http.ResponseWriter, r *http.Request) {
	// Database connection string
	connStr := fmt.Sprintf("host=%s port=5439 user=%s password=%s dbname=dev sslmode=require",
		os.Getenv("redshift_db_host"),
		os.Getenv("redshift_db_user"),
		os.Getenv("redshift_db_password"))

	// Connect to database
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		http.Error(w, "Database connection failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer db.Close()

	// Test connection
	if err := db.Ping(); err != nil {
		http.Error(w, "Database ping failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Set content type
	w.Header().Set("Content-Type", "text/html")

	// Write HTML response
	fmt.Fprintf(w, `
<!DOCTYPE html>
<html>
<head>
    <title>Redshift Column Names</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        ul { list-style-type: none; padding: 0; }
        li { background: #f4f4f4; margin: 5px 0; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Product Pairs Table - Column Names</h1>
`)

	// First, let's see what tables exist
	debugQuery := `SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';`
	debugRows, err := db.Query(debugQuery)
	if err != nil {
		http.Error(w, "Debug query failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintf(w, "<h2>Available tables:</h2><ul>")
	for debugRows.Next() {
		var tableName string
		debugRows.Scan(&tableName)
		fmt.Fprintf(w, "<li>%s</li>", tableName)
	}
	fmt.Fprintf(w, "</ul>")
	debugRows.Close()

	// Query to get column names
	query := `
		SELECT column_name 
		FROM information_schema.columns 
		WHERE table_name = 'product_pairs' 
		ORDER BY ordinal_position;
	`

	rows, err := db.Query(query)
	if err != nil {
		http.Error(w, "Query failed: "+err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	fmt.Fprintf(w, "<h2>Product Pairs Columns:</h2><ul>")

	// Fetch and display column names
	for rows.Next() {
		var columnName string
		if err := rows.Scan(&columnName); err != nil {
			http.Error(w, "Row scan failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		fmt.Fprintf(w, "        <li>%s</li>\n", columnName)
	}

	fmt.Fprintf(w, `
    </ul>
    <p><em>Connected to Redshift successfully!</em></p>
</body>
</html>
`)

	// Check for errors during iteration
	if err := rows.Err(); err != nil {
		log.Printf("Row iteration error: %v", err)
	}
}
