#!/bin/bash

echo "Setting up RAG system with Prisma and SQLite..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Generate Prisma client
echo "Generating Prisma client..."
python -m prisma generate

# Push database schema
echo "Creating database schema..."
python -m prisma db push

echo "Setup complete!"
echo ""
echo "To add sample documents, run: python add_documents.py"
echo "To run the main application, run: python main.py"
