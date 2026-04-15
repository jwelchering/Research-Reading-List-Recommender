<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Research-Reading-List-Recommender
Building AI course project

## Summary

**Research Reading List Recommender** is an AI-based tool that helps researchers organize notes and discover relevant sources or topic clusters. It takes short notes, keywords, abstracts, or paper titles as input and recommends related literature and thematic groups. The aim is to make research work more structured and reduce the time spent manually sorting references.

## Background

Researchers often collect many notes, paper titles, and abstracts, but it becomes difficult to keep track of what belongs together. Valuable sources can be overlooked simply because they are not yet organized into themes or compared with related material. This project addresses the problem of fragmented research notes by suggesting relevant sources and grouping similar ideas automatically.

## How is it used?

Our solution uses text similarity and clustering to connect notes with relevant research material. It can suggest similar papers, detect related themes, and help users see patterns across their reading list. In this way, the tool supports faster literature review and more structured research planning.

## The idea

The idea is to build a lightweight recommendation system for research notes and academic sources. A user can enter notes, paper titles, or short text excerpts, and the system will compare them with a collection of stored documents to find the most relevant matches. In addition, it can group similar notes into topic clusters so that the user can identify recurring themes across their work. The core approach can be implemented with natural language processing methods such as TF-IDF, cosine similarity, and simple clustering techniques. This makes the project practical, understandable, and realistic to implement within a student project context.

## Project development roadmap

The project can start with a small dataset of research notes or abstracts from one subject area, such as theology, AI ethics, or digital transformation. The first version will clean and vectorize the text, then compute similarities and return the top recommended sources for each note. A second step will add topic clustering to group related material more clearly. Later improvements could include a simple interface, keyword highlighting, and support for larger document collections.

## Possible future extensions

- Add a web interface for uploading notes and viewing recommendations.
- Support larger document collections and multiple subject areas.
- Highlight matching keywords inside recommended sources.
- Add user feedback so the system can improve its recommendations over time.
- Combine text similarity with metadata such as author, year, or topic.

## Technologies

- Python
- NumPy
- scikit-learn
- Natural Language Processing
- TF-IDF
- Cosine Similarity
- Clustering

## How to use

1. Add your research notes or source texts to the dataset.
2. Run the preprocessing step to clean and tokenize the text.
3. Convert the documents into TF-IDF vectors.
4. Calculate similarity between the input note and stored sources.
5. Display the most relevant sources and topic clusters.

## Project goal

The goal of this project is to support academic reading and note organization with a simple but useful AI tool. It is designed to be understandable, extendable, and useful for students and researchers who work with many texts.

## License

This project is intended for educational use.
