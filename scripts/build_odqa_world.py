"""Build the offline open-domain QA "world" used by the UG-RAV simulator and demo.

Outputs ``data/odqa/world.jsonl`` (questions) and ``data/odqa/passages.jsonl``
(retrieval corpus). Each question carries gold aliases, an answer type (for
distractor sampling), a popularity tier (a judgement-based proxy for Wikipedia
page views, mirroring PopQA's long-tail analysis), an optional common
misconception (TruthfulQA-style imitative falsehood) and a gold evidence
passage. About 12 % of gold passages are withheld from the corpus to simulate
retrieval failure, so retrieval can mislead (external hallucination).

All facts were written for this project and double-checked; they are used for
simulation and demonstration, not as a benchmark of real LLMs (see
``scripts/run_real_experiment.py`` for the real-model experiment on PopQA).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Books: (title, descriptor, author, aliases, tier, year or "", misconception or None)
BOOKS = [
    ("Godan", "Hindi novel", "Munshi Premchand", ["Premchand", "Munshi Premchand", "Dhanpat Rai Srivastava"], "mid", "1936", "Rabindranath Tagore"),
    ("Gaban", "Hindi novel", "Munshi Premchand", ["Premchand", "Munshi Premchand"], "low", "1931", None),
    ("Gitanjali", "collection of poems", "Rabindranath Tagore", ["Tagore", "Rabindranath Tagore"], "high", "1910", None),
    ("Gora", "Bengali novel", "Rabindranath Tagore", ["Tagore", "Rabindranath Tagore"], "low", "1910", None),
    ("Midnight's Children", "novel", "Salman Rushdie", ["Salman Rushdie", "Rushdie"], "high", "1981", None),
    ("The God of Small Things", "novel", "Arundhati Roy", ["Arundhati Roy"], "high", "1997", None),
    ("Malgudi Days", "short-story collection", "R. K. Narayan", ["R. K. Narayan", "R.K. Narayan", "Narayan"], "mid", "1943", None),
    ("The Guide", "novel", "R. K. Narayan", ["R. K. Narayan", "R.K. Narayan", "Narayan"], "mid", "1958", None),
    ("Devdas", "Bengali novel", "Sarat Chandra Chattopadhyay", ["Sarat Chandra Chattopadhyay", "Sarat Chandra Chatterjee", "Saratchandra Chattopadhyay"], "mid", "1917", None),
    ("Anandamath", "Bengali novel", "Bankim Chandra Chattopadhyay", ["Bankim Chandra Chattopadhyay", "Bankim Chandra Chatterjee", "Bankim Chandra"], "mid", "1882", None),
    ("Train to Pakistan", "novel", "Khushwant Singh", ["Khushwant Singh"], "mid", "1956", None),
    ("A Suitable Boy", "novel", "Vikram Seth", ["Vikram Seth"], "mid", "1993", None),
    ("The White Tiger", "novel", "Aravind Adiga", ["Aravind Adiga", "Adiga"], "mid", "2008", None),
    ("The Inheritance of Loss", "novel", "Kiran Desai", ["Kiran Desai"], "low", "2006", None),
    ("Madhushala", "Hindi poetry collection", "Harivansh Rai Bachchan", ["Harivansh Rai Bachchan", "Bachchan"], "mid", "1935", None),
    ("Kamayani", "Hindi epic poem", "Jaishankar Prasad", ["Jaishankar Prasad"], "low", "1936", None),
    ("Raag Darbari", "Hindi satirical novel", "Shrilal Shukla", ["Shrilal Shukla"], "low", "1968", None),
    ("Tamas", "Hindi novel", "Bhisham Sahni", ["Bhisham Sahni"], "low", "", None),
    ("Chitralekha", "Hindi novel", "Bhagwati Charan Verma", ["Bhagwati Charan Verma"], "low", "1934", None),
    ("Maila Anchal", "Hindi novel", "Phanishwar Nath Renu", ["Phanishwar Nath Renu", "Phanishwarnath Renu", "Renu"], "low", "1954", None),
    ("Pather Panchali", "Bengali novel", "Bibhutibhushan Bandyopadhyay", ["Bibhutibhushan Bandyopadhyay", "Bibhutibhushan Banerjee"], "low", "1929", "Satyajit Ray"),
    ("Chemmeen", "Malayalam novel", "Thakazhi Sivasankara Pillai", ["Thakazhi Sivasankara Pillai", "Thakazhi"], "low", "1956", None),
    ("Samskara", "Kannada novel", "U. R. Ananthamurthy", ["U. R. Ananthamurthy", "U.R. Ananthamurthy", "Ananthamurthy"], "low", "1965", None),
    ("Ponniyin Selvan", "Tamil historical novel", "Kalki Krishnamurthy", ["Kalki Krishnamurthy", "Kalki"], "mid", "", None),
    ("Mrityunjay", "Marathi novel", "Shivaji Sawant", ["Shivaji Sawant"], "low", "1967", None),
    ("Yayati", "Marathi novel", "V. S. Khandekar", ["V. S. Khandekar", "Vishnu Sakharam Khandekar", "Khandekar"], "low", "", None),
    ("Pride and Prejudice", "novel", "Jane Austen", ["Jane Austen", "Austen"], "high", "1813", None),
    ("Nineteen Eighty-Four", "novel", "George Orwell", ["George Orwell", "Orwell", "Eric Arthur Blair"], "high", "1949", None),
    ("War and Peace", "novel", "Leo Tolstoy", ["Leo Tolstoy", "Tolstoy"], "high", "1869", None),
    ("One Hundred Years of Solitude", "novel", "Gabriel García Márquez", ["Gabriel García Márquez", "Gabriel Garcia Marquez", "García Márquez", "Marquez"], "high", "1967", None),
    ("The Old Man and the Sea", "novella", "Ernest Hemingway", ["Ernest Hemingway", "Hemingway"], "high", "1952", None),
    ("To Kill a Mockingbird", "novel", "Harper Lee", ["Harper Lee"], "high", "1960", None),
    ("Things Fall Apart", "novel", "Chinua Achebe", ["Chinua Achebe", "Achebe"], "mid", "1958", None),
    ("Crime and Punishment", "novel", "Fyodor Dostoevsky", ["Fyodor Dostoevsky", "Dostoevsky", "Dostoyevsky"], "high", "1866", None),
    ("Don Quixote", "novel", "Miguel de Cervantes", ["Miguel de Cervantes", "Cervantes"], "high", "", None),
    ("The Tale of Genji", "classic Japanese novel", "Murasaki Shikibu", ["Murasaki Shikibu", "Lady Murasaki"], "mid", "", None),
    ("Abhijnanashakuntalam", "Sanskrit play", "Kalidasa", ["Kalidasa", "Kālidāsa"], "mid", "", None),
    ("Meghaduta", "Sanskrit lyric poem", "Kalidasa", ["Kalidasa", "Kālidāsa"], "low", "", None),
    ("Arthashastra", "ancient Indian treatise on statecraft", "Kautilya", ["Kautilya", "Chanakya", "Vishnugupta"], "mid", "", None),
    ("Ramcharitmanas", "Awadhi epic poem", "Tulsidas", ["Tulsidas", "Goswami Tulsidas"], "mid", "", "Valmiki"),
    ("Wings of Fire", "autobiography", "A. P. J. Abdul Kalam", ["A. P. J. Abdul Kalam", "APJ Abdul Kalam", "Abdul Kalam", "Kalam"], "mid", "1999", None),
    ("The Discovery of India", "book", "Jawaharlal Nehru", ["Jawaharlal Nehru", "Nehru"], "mid", "1946", None),
    ("Hind Swaraj", "book", "Mahatma Gandhi", ["Mahatma Gandhi", "Gandhi", "Mohandas Karamchand Gandhi", "M. K. Gandhi"], "mid", "1909", None),
    ("Annihilation of Caste", "essay", "B. R. Ambedkar", ["B. R. Ambedkar", "Ambedkar", "Bhimrao Ambedkar", "Babasaheb Ambedkar"], "mid", "1936", None),
    ("Lilavati", "mathematical treatise", "Bhaskara II", ["Bhaskara II", "Bhaskaracharya"], "low", "", None),
    ("Aryabhatiya", "astronomical treatise", "Aryabhata", ["Aryabhata", "Aryabhatta"], "mid", "", None),
    ("Ashtadhyayi", "Sanskrit grammar", "Panini", ["Panini", "Pāṇini"], "mid", "", None),
    ("Yoga Sutras", "Sanskrit text on yoga", "Patanjali", ["Patanjali"], "mid", "", None),
    ("Sushruta Samhita", "Sanskrit text on medicine and surgery", "Sushruta", ["Sushruta"], "low", "", None),
    ("Charaka Samhita", "Sanskrit text on Ayurveda", "Charaka", ["Charaka"], "low", "", None),
    ("Brahmasphutasiddhanta", "astronomical treatise", "Brahmagupta", ["Brahmagupta"], "low", "", None),
]

# Capitals: (place, kind, capital, aliases, tier, misconception or None)
CAPITALS = [
    ("Australia", "country", "Canberra", ["Canberra"], "high", "Sydney"),
    ("Canada", "country", "Ottawa", ["Ottawa"], "high", "Toronto"),
    ("Turkey", "country", "Ankara", ["Ankara"], "high", "Istanbul"),
    ("Brazil", "country", "Brasília", ["Brasília", "Brasilia"], "high", "Rio de Janeiro"),
    ("Nigeria", "country", "Abuja", ["Abuja"], "mid", "Lagos"),
    ("Switzerland", "country", "Bern", ["Bern", "Berne"], "mid", "Zurich"),
    ("Morocco", "country", "Rabat", ["Rabat"], "mid", "Casablanca"),
    ("Tanzania", "country", "Dodoma", ["Dodoma"], "low", "Dar es Salaam"),
    ("Myanmar", "country", "Naypyidaw", ["Naypyidaw", "Nay Pyi Taw", "Naypyitaw"], "mid", "Yangon"),
    ("Kazakhstan", "country", "Astana", ["Astana", "Nur-Sultan"], "mid", "Almaty"),
    ("New Zealand", "country", "Wellington", ["Wellington"], "high", "Auckland"),
    ("Vietnam", "country", "Hanoi", ["Hanoi"], "mid", "Ho Chi Minh City"),
    ("Pakistan", "country", "Islamabad", ["Islamabad"], "high", "Karachi"),
    ("Bhutan", "country", "Thimphu", ["Thimphu"], "mid", None),
    ("Mongolia", "country", "Ulaanbaatar", ["Ulaanbaatar", "Ulan Bator"], "mid", None),
    ("Japan", "country", "Tokyo", ["Tokyo"], "high", None),
    ("France", "country", "Paris", ["Paris"], "high", None),
    ("Egypt", "country", "Cairo", ["Cairo"], "high", None),
    ("Kenya", "country", "Nairobi", ["Nairobi"], "mid", None),
    ("Peru", "country", "Lima", ["Lima"], "mid", None),
    ("Bangladesh", "country", "Dhaka", ["Dhaka"], "high", None),
    ("Nepal", "country", "Kathmandu", ["Kathmandu"], "high", None),
    ("Arunachal Pradesh", "Indian state", "Itanagar", ["Itanagar"], "low", None),
    ("Mizoram", "Indian state", "Aizawl", ["Aizawl"], "low", None),
    ("Nagaland", "Indian state", "Kohima", ["Kohima"], "low", "Dimapur"),
    ("Meghalaya", "Indian state", "Shillong", ["Shillong"], "mid", None),
    ("Manipur", "Indian state", "Imphal", ["Imphal"], "mid", None),
    ("Sikkim", "Indian state", "Gangtok", ["Gangtok"], "mid", None),
    ("Tripura", "Indian state", "Agartala", ["Agartala"], "low", None),
    ("Chhattisgarh", "Indian state", "Raipur", ["Raipur", "Nava Raipur"], "mid", None),
    ("Jharkhand", "Indian state", "Ranchi", ["Ranchi"], "mid", "Jamshedpur"),
    ("Assam", "Indian state", "Dispur", ["Dispur"], "mid", None),
    ("Odisha", "Indian state", "Bhubaneswar", ["Bhubaneswar"], "mid", "Cuttack"),
    ("Goa", "Indian state", "Panaji", ["Panaji", "Panjim"], "mid", "Margao"),
    ("Kerala", "Indian state", "Thiruvananthapuram", ["Thiruvananthapuram", "Trivandrum"], "mid", "Kochi"),
    ("Madhya Pradesh", "Indian state", "Bhopal", ["Bhopal"], "mid", "Indore"),
    ("Gujarat", "Indian state", "Gandhinagar", ["Gandhinagar"], "mid", "Ahmedabad"),
    ("Karnataka", "Indian state", "Bengaluru", ["Bengaluru", "Bangalore"], "high", None),
    ("Telangana", "Indian state", "Hyderabad", ["Hyderabad"], "high", None),
    ("Rajasthan", "Indian state", "Jaipur", ["Jaipur"], "high", None),
    ("Bihar", "Indian state", "Patna", ["Patna"], "high", None),
    ("Maharashtra", "Indian state", "Mumbai", ["Mumbai", "Bombay"], "high", "Pune"),
    ("Tamil Nadu", "Indian state", "Chennai", ["Chennai", "Madras"], "high", None),
    ("West Bengal", "Indian state", "Kolkata", ["Kolkata", "Calcutta"], "high", None),
    ("Uttar Pradesh", "Indian state", "Lucknow", ["Lucknow"], "high", None),
]

# Free-form facts: (group, answer_type, tier, question, answer, aliases, passage, misconception or None)
FACTS = [
    # monuments and architecture
    ("monument", "person", "high", "Who built the Taj Mahal?", "Shah Jahan", ["Shah Jahan", "Shahjahan"],
     "The Taj Mahal in Agra was commissioned by the Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal.", "Akbar"),
    ("monument", "person", "high", "Which Mughal emperor built the Red Fort in Delhi?", "Shah Jahan", ["Shah Jahan", "Shahjahan"],
     "The Red Fort in Delhi was built by the Mughal emperor Shah Jahan as the palace of his capital Shahjahanabad.", "Akbar"),
    ("monument", "person", "mid", "Which Mughal emperor founded Fatehpur Sikri?", "Akbar", ["Akbar", "Akbar the Great"],
     "Fatehpur Sikri, near Agra, was founded by the Mughal emperor Akbar in the sixteenth century.", None),
    ("monument", "person", "high", "Who began the construction of the Qutub Minar?", "Qutb ud-Din Aibak", ["Qutb ud-Din Aibak", "Qutbuddin Aibak", "Qutb-ud-din Aibak", "Aibak"],
     "Construction of the Qutub Minar in Delhi was begun by Qutb ud-Din Aibak and continued by his successor Iltutmish.", "Iltutmish"),
    ("monument", "person", "mid", "Who built the Charminar in Hyderabad?", "Muhammad Quli Qutb Shah", ["Muhammad Quli Qutb Shah", "Quli Qutb Shah"],
     "The Charminar in Hyderabad was built in 1591 by Muhammad Quli Qutb Shah of the Qutb Shahi dynasty.", None),
    ("monument", "person", "mid", "Which Chola king built the Brihadeeswarar Temple at Thanjavur?", "Rajaraja Chola I", ["Rajaraja Chola I", "Rajaraja Chola", "Rajaraja I", "Raja Raja Chola"],
     "The Brihadeeswarar Temple at Thanjavur was built by the Chola king Rajaraja Chola I in the early eleventh century.", None),
    ("monument", "person", "low", "Which Eastern Ganga king built the Konark Sun Temple?", "Narasimhadeva I", ["Narasimhadeva I", "Narasimha Deva I", "Narasimhadeva"],
     "The Konark Sun Temple in Odisha was built around 1250 by Narasimhadeva I of the Eastern Ganga dynasty.", None),
    ("monument", "person", "mid", "Which emperor originally commissioned the Great Stupa at Sanchi?", "Ashoka", ["Ashoka", "Asoka", "Emperor Ashoka", "Ashoka the Great"],
     "The Great Stupa at Sanchi was originally commissioned by the Mauryan emperor Ashoka in the third century BCE.", None),
    ("monument", "person", "mid", "Who built the Hawa Mahal in Jaipur?", "Sawai Pratap Singh", ["Sawai Pratap Singh", "Maharaja Sawai Pratap Singh", "Pratap Singh"],
     "The Hawa Mahal in Jaipur was built in 1799 by Maharaja Sawai Pratap Singh.", "Sawai Jai Singh II"),
    ("monument", "person", "low", "Who built the Jantar Mantar observatory in Jaipur?", "Sawai Jai Singh II", ["Sawai Jai Singh II", "Jai Singh II", "Sawai Jai Singh"],
     "The Jantar Mantar astronomical observatory in Jaipur was built by Maharaja Sawai Jai Singh II.", None),
    ("monument", "person", "low", "Who commissioned Humayun's Tomb in Delhi?", "Bega Begum", ["Bega Begum", "Haji Begum"],
     "Humayun's Tomb in Delhi was commissioned by Bega Begum, the chief consort of the Mughal emperor Humayun.", "Humayun"),
    ("monument", "person", "low", "Which architect designed the Gateway of India?", "George Wittet", ["George Wittet", "Wittet"],
     "The Gateway of India in Mumbai was designed by the Scottish architect George Wittet.", None),
    ("monument", "person", "high", "Whose engineering company designed and built the Eiffel Tower?", "Gustave Eiffel", ["Gustave Eiffel", "Eiffel"],
     "The Eiffel Tower in Paris was designed and built by the engineering company of Gustave Eiffel.", None),
    ("monument", "person", "low", "Which architect designed the Victoria Memorial in Kolkata?", "William Emerson", ["William Emerson", "Sir William Emerson", "Emerson"],
     "The Victoria Memorial in Kolkata was designed by the architect Sir William Emerson.", None),
    ("monument", "person", "mid", "Under which Sikh guru was the Harmandir Sahib in Amritsar built?", "Guru Arjan", ["Guru Arjan", "Guru Arjan Dev", "Arjan Dev"],
     "The Harmandir Sahib, the Golden Temple in Amritsar, was built under the fifth Sikh guru, Guru Arjan.", "Guru Nanak"),
    # science and invention
    ("science", "person", "high", "Who discovered penicillin?", "Alexander Fleming", ["Alexander Fleming", "Fleming"],
     "Penicillin was discovered in 1928 by the Scottish physician Alexander Fleming.", None),
    ("science", "person", "mid", "Who discovered the Raman effect?", "C. V. Raman", ["C. V. Raman", "C.V. Raman", "Raman", "Chandrasekhara Venkata Raman"],
     "The Raman effect, the inelastic scattering of light, was discovered in 1928 by the Indian physicist C. V. Raman.", None),
    ("science", "person", "high", "Who is credited with inventing the telephone?", "Alexander Graham Bell", ["Alexander Graham Bell", "Graham Bell", "Bell"],
     "Alexander Graham Bell was granted the first United States patent for the telephone in 1876.", "Thomas Edison"),
    ("science", "person", "high", "Who invented the World Wide Web?", "Tim Berners-Lee", ["Tim Berners-Lee", "Berners-Lee"],
     "The World Wide Web was invented in 1989 by the British scientist Tim Berners-Lee at CERN.", "Bill Gates"),
    ("science", "person", "high", "Who invented dynamite?", "Alfred Nobel", ["Alfred Nobel", "Nobel"],
     "Dynamite was invented in 1867 by the Swedish chemist Alfred Nobel.", None),
    ("science", "person", "low", "Who invented the crescograph?", "Jagadish Chandra Bose", ["Jagadish Chandra Bose", "J. C. Bose", "Jagadish Bose"],
     "The crescograph, an instrument for measuring plant growth, was invented by Jagadish Chandra Bose.", "Satyendra Nath Bose"),
    ("science", "person", "mid", "Which Indian physicist's work led to Bose–Einstein statistics?", "Satyendra Nath Bose", ["Satyendra Nath Bose", "S. N. Bose", "Satyendranath Bose"],
     "Bose–Einstein statistics grew out of the work of the Indian physicist Satyendra Nath Bose, extended by Albert Einstein.", "Jagadish Chandra Bose"),
    ("science", "person", "low", "After whom is the Chandrasekhar limit named?", "Subrahmanyan Chandrasekhar", ["Subrahmanyan Chandrasekhar", "S. Chandrasekhar", "Chandrasekhar"],
     "The Chandrasekhar limit, the maximum mass of a stable white dwarf star, is named after Subrahmanyan Chandrasekhar.", None),
    ("science", "person", "high", "Who developed the first smallpox vaccine?", "Edward Jenner", ["Edward Jenner", "Jenner"],
     "The first smallpox vaccine was developed in 1796 by the English physician Edward Jenner.", "Louis Pasteur"),
    ("science", "person", "mid", "Who discovered radioactivity?", "Henri Becquerel", ["Henri Becquerel", "Becquerel", "Antoine Henri Becquerel"],
     "Radioactivity was discovered in 1896 by the French physicist Henri Becquerel.", "Marie Curie"),
    ("science", "person", "high", "Who published the first widely recognised periodic table in 1869?", "Dmitri Mendeleev", ["Dmitri Mendeleev", "Mendeleev", "Mendeleyev"],
     "In 1869 the Russian chemist Dmitri Mendeleev published the first widely recognised periodic table of the elements.", None),
    ("science", "person", "mid", "Who discovered the electron?", "J. J. Thomson", ["J. J. Thomson", "J.J. Thomson", "Thomson", "Joseph John Thomson"],
     "The electron was discovered in 1897 by the British physicist J. J. Thomson.", "Ernest Rutherford"),
    ("science", "person", "mid", "Who discovered the neutron?", "James Chadwick", ["James Chadwick", "Chadwick"],
     "The neutron was discovered in 1932 by the British physicist James Chadwick.", "Ernest Rutherford"),
    ("science", "person", "high", "Who introduced printing with movable metal type in Europe?", "Johannes Gutenberg", ["Johannes Gutenberg", "Gutenberg"],
     "Johannes Gutenberg introduced printing with movable metal type in Europe around 1440.", None),
    ("science", "person", "mid", "Who co-discovered insulin with Charles Best in 1921?", "Frederick Banting", ["Frederick Banting", "Banting", "Sir Frederick Banting"],
     "Insulin was discovered in 1921 by Frederick Banting and Charles Best at the University of Toronto.", None),
    ("science", "person", "high", "Who proposed the heliocentric model of the Solar System in Renaissance Europe?", "Nicolaus Copernicus", ["Nicolaus Copernicus", "Copernicus"],
     "Nicolaus Copernicus proposed the heliocentric model, placing the Sun at the centre of the Solar System.", "Galileo Galilei"),
    ("science", "person", "high", "Who formulated the general theory of relativity?", "Albert Einstein", ["Albert Einstein", "Einstein"],
     "The general theory of relativity was formulated by Albert Einstein in 1915.", None),
    ("science", "person", "low", "Which Indian-born scientist shared the 1968 Nobel Prize in Physiology or Medicine?", "Har Gobind Khorana", ["Har Gobind Khorana", "Hargobind Khorana", "Khorana"],
     "Har Gobind Khorana shared the 1968 Nobel Prize in Physiology or Medicine for work on the genetic code.", None),
    ("science", "person", "mid", "Which mathematician is associated with the taxicab number 1729?", "Srinivasa Ramanujan", ["Srinivasa Ramanujan", "Ramanujan"],
     "The number 1729 is known as the Hardy–Ramanujan number after an anecdote about the mathematician Srinivasa Ramanujan.", None),
    # people and firsts
    ("people", "person", "high", "Who was the first President of India?", "Rajendra Prasad", ["Rajendra Prasad", "Dr. Rajendra Prasad"],
     "Rajendra Prasad served as the first President of India from 1950 to 1962.", "Jawaharlal Nehru"),
    ("people", "person", "high", "Who was the first Prime Minister of India?", "Jawaharlal Nehru", ["Jawaharlal Nehru", "Nehru"],
     "Jawaharlal Nehru was the first Prime Minister of India, serving from 1947 until 1964.", None),
    ("people", "person", "high", "Who was the first woman Prime Minister of India?", "Indira Gandhi", ["Indira Gandhi"],
     "Indira Gandhi became the first woman Prime Minister of India in 1966.", None),
    ("people", "person", "high", "Who was the first Indian to win a Nobel Prize?", "Rabindranath Tagore", ["Rabindranath Tagore", "Tagore"],
     "Rabindranath Tagore became the first Indian Nobel laureate when he won the Nobel Prize in Literature in 1913.", "C. V. Raman"),
    ("people", "person", "mid", "Who was the first Indian citizen to travel to space?", "Rakesh Sharma", ["Rakesh Sharma"],
     "Rakesh Sharma became the first Indian citizen in space in 1984 aboard Soyuz T-11.", "Kalpana Chawla"),
    ("people", "person", "mid", "Who was the first woman of Indian origin to go to space?", "Kalpana Chawla", ["Kalpana Chawla"],
     "Kalpana Chawla, who flew on the Space Shuttle Columbia in 1997, was the first woman of Indian origin in space.", "Sunita Williams"),
    ("people", "person", "high", "Who wrote the Indian national anthem Jana Gana Mana?", "Rabindranath Tagore", ["Rabindranath Tagore", "Tagore"],
     "Jana Gana Mana, the national anthem of India, was written by Rabindranath Tagore.", "Bankim Chandra Chattopadhyay"),
    ("people", "person", "mid", "Who wrote Vande Mataram?", "Bankim Chandra Chattopadhyay", ["Bankim Chandra Chattopadhyay", "Bankim Chandra Chatterjee", "Bankim Chandra"],
     "Vande Mataram, the national song of India, was written by Bankim Chandra Chattopadhyay and appears in his novel Anandamath.", "Rabindranath Tagore"),
    ("people", "person", "mid", "Who wrote the poem Saare Jahan Se Achha?", "Muhammad Iqbal", ["Muhammad Iqbal", "Allama Iqbal", "Iqbal"],
     "The poem Saare Jahan Se Achha was written by the poet Muhammad Iqbal.", None),
    ("people", "person", "high", "Who founded the Mughal Empire in India?", "Babur", ["Babur", "Zahir-ud-din Muhammad Babur"],
     "Babur founded the Mughal Empire in India after his victory at the First Battle of Panipat in 1526.", "Akbar"),
    ("people", "person", "high", "Who founded the Maurya Empire?", "Chandragupta Maurya", ["Chandragupta Maurya", "Chandragupta"],
     "The Maurya Empire was founded by Chandragupta Maurya with the help of his adviser Chanakya.", "Ashoka"),
    ("people", "person", "high", "Who founded Sikhism?", "Guru Nanak", ["Guru Nanak", "Guru Nanak Dev", "Nanak"],
     "Sikhism was founded in the Punjab by Guru Nanak, the first of the ten Sikh gurus.", None),
    ("people", "person", "mid", "Who was the twenty-fourth Tirthankara of Jainism?", "Mahavira", ["Mahavira", "Vardhamana", "Mahavir"],
     "Mahavira, born Vardhamana, was the twenty-fourth and last Tirthankara of Jainism.", "Parshvanatha"),
    ("people", "person", "high", "Who chaired the Drafting Committee of the Constitution of India?", "B. R. Ambedkar", ["B. R. Ambedkar", "Ambedkar", "Bhimrao Ambedkar", "Babasaheb Ambedkar"],
     "B. R. Ambedkar chaired the Drafting Committee of the Constituent Assembly that framed the Constitution of India.", "Jawaharlal Nehru"),
    ("people", "person", "high", "Who is known as the Iron Man of India?", "Vallabhbhai Patel", ["Vallabhbhai Patel", "Sardar Patel", "Sardar Vallabhbhai Patel"],
     "Sardar Vallabhbhai Patel is known as the Iron Man of India for unifying the princely states.", None),
    ("people", "person", "mid", "Who is regarded as the father of the Indian space programme?", "Vikram Sarabhai", ["Vikram Sarabhai", "Sarabhai"],
     "Vikram Sarabhai is regarded as the father of the Indian space programme and founded ISRO.", "A. P. J. Abdul Kalam"),
    ("people", "person", "low", "Who was the first Indian Governor-General of independent India?", "C. Rajagopalachari", ["C. Rajagopalachari", "Rajagopalachari", "Rajaji", "Chakravarti Rajagopalachari"],
     "C. Rajagopalachari succeeded Lord Mountbatten in 1948 as the first and only Indian Governor-General of India.", "Lord Mountbatten"),
    ("people", "person", "high", "Who directed the film Pather Panchali?", "Satyajit Ray", ["Satyajit Ray", "Ray"],
     "The 1955 film Pather Panchali was directed by Satyajit Ray.", None),
    ("people", "person", "mid", "Who directed the film Sholay?", "Ramesh Sippy", ["Ramesh Sippy"],
     "The 1975 film Sholay was directed by Ramesh Sippy.", "Yash Chopra"),
    ("people", "person", "low", "Who directed the film Mother India?", "Mehboob Khan", ["Mehboob Khan"],
     "The 1957 film Mother India was directed by Mehboob Khan.", "Raj Kapoor"),
    ("people", "person", "high", "Who composed the music for the film Roja?", "A. R. Rahman", ["A. R. Rahman", "AR Rahman", "Rahman"],
     "The music for the 1992 film Roja was composed by A. R. Rahman.", "Ilaiyaraaja"),
    ("people", "person", "mid", "Which sitar maestro taught George Harrison?", "Ravi Shankar", ["Ravi Shankar", "Pandit Ravi Shankar"],
     "The sitar maestro Ravi Shankar taught the sitar to George Harrison of the Beatles.", None),
    ("people", "person", "high", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["Leonardo da Vinci", "Leonardo", "da Vinci"],
     "The Mona Lisa was painted by Leonardo da Vinci in the early sixteenth century.", None),
    ("people", "person", "high", "Who painted The Starry Night?", "Vincent van Gogh", ["Vincent van Gogh", "Van Gogh"],
     "The Starry Night was painted in 1889 by Vincent van Gogh.", None),
    ("people", "person", "low", "Who painted Bharat Mata in 1905?", "Abanindranath Tagore", ["Abanindranath Tagore"],
     "The painting Bharat Mata of 1905 is the work of Abanindranath Tagore.", "Rabindranath Tagore"),
    ("people", "person", "mid", "Who was the first Indian to win an individual Olympic gold medal?", "Abhinav Bindra", ["Abhinav Bindra", "Bindra"],
     "Abhinav Bindra won the 10 metre air rifle event at the 2008 Beijing Olympics, India's first individual Olympic gold.", "Neeraj Chopra"),
    ("people", "person", "mid", "Which Indian hockey player is known as The Wizard?", "Dhyan Chand", ["Dhyan Chand", "Major Dhyan Chand"],
     "The Indian hockey player Dhyan Chand is known as The Wizard for his ball control.", None),
    ("people", "person", "high", "Who captained India to victory in the 1983 Cricket World Cup?", "Kapil Dev", ["Kapil Dev"],
     "Kapil Dev captained India to victory in the 1983 Cricket World Cup.", "Sunil Gavaskar"),
    ("people", "person", "low", "Who was the first Indian woman to win an Olympic medal?", "Karnam Malleswari", ["Karnam Malleswari", "Malleswari"],
     "The weightlifter Karnam Malleswari won bronze at the 2000 Sydney Olympics, the first Olympic medal by an Indian woman.", "P. T. Usha"),
    ("people", "person", "mid", "Who became India's first chess grandmaster?", "Viswanathan Anand", ["Viswanathan Anand", "Vishy Anand", "Anand"],
     "Viswanathan Anand became India's first chess grandmaster in 1988.", None),
    ("people", "person", "high", "Who scored the first double century in men's One Day International cricket?", "Sachin Tendulkar", ["Sachin Tendulkar", "Tendulkar"],
     "Sachin Tendulkar scored the first double century in men's One Day International cricket in 2010.", "Virender Sehwag"),
    ("people", "person", "high", "Who led the Salt March in 1930?", "Mahatma Gandhi", ["Mahatma Gandhi", "Gandhi", "Mohandas Karamchand Gandhi"],
     "Mahatma Gandhi led the Salt March from Sabarmati to Dandi in 1930.", None),
    # years
    ("history", "year", "mid", "In which year was the Battle of Plassey fought?", "1757", ["1757"],
     "The Battle of Plassey was fought in 1757 between the East India Company and the Nawab of Bengal.", "1764"),
    ("history", "year", "mid", "In which year was the First Battle of Panipat fought?", "1526", ["1526"],
     "The First Battle of Panipat was fought in 1526 between Babur and Ibrahim Lodi.", "1556"),
    ("history", "year", "high", "In which year did India gain independence?", "1947", ["1947"],
     "India gained independence from British rule on 15 August 1947.", None),
    ("history", "year", "high", "In which year did the Constitution of India come into effect?", "1950", ["1950"],
     "The Constitution of India came into effect on 26 January 1950.", "1949"),
    ("history", "year", "high", "In which year did the Jallianwala Bagh massacre take place?", "1919", ["1919"],
     "The Jallianwala Bagh massacre took place in Amritsar in 1919.", None),
    ("history", "year", "mid", "In which year was the Quit India Movement launched?", "1942", ["1942"],
     "The Quit India Movement was launched by Mahatma Gandhi in August 1942.", None),
    ("history", "year", "mid", "In which year did Chandrayaan-3 land on the Moon?", "2023", ["2023"],
     "Chandrayaan-3 landed near the south pole of the Moon in August 2023.", "2019"),
    ("history", "year", "low", "In which year was the Indian Space Research Organisation founded?", "1969", ["1969"],
     "The Indian Space Research Organisation was founded on 15 August 1969.", "1962"),
    ("history", "year", "low", "In which year did C. V. Raman receive the Nobel Prize in Physics?", "1930", ["1930"],
     "C. V. Raman received the Nobel Prize in Physics in 1930.", "1928"),
    # geography and science facts
    ("geography", "river", "mid", "On which river is the Bhakra Dam built?", "Sutlej", ["Sutlej", "Satluj", "Sutlej River"],
     "The Bhakra Dam is built across the Sutlej river in Himachal Pradesh.", "Beas"),
    ("geography", "river", "mid", "On which river is the Hirakud Dam built?", "Mahanadi", ["Mahanadi", "Mahanadi River"],
     "The Hirakud Dam is built across the Mahanadi river in Odisha.", None),
    ("geography", "river", "mid", "On which river is the Sardar Sarovar Dam built?", "Narmada", ["Narmada", "Narmada River"],
     "The Sardar Sarovar Dam is built on the Narmada river in Gujarat.", "Tapti"),
    ("geography", "river", "low", "Which river flows through Srinagar?", "Jhelum", ["Jhelum", "Jhelum River"],
     "The Jhelum river flows through the city of Srinagar in Kashmir.", "Indus"),
    ("geography", "state", "low", "In which Indian state is Loktak Lake?", "Manipur", ["Manipur"],
     "Loktak Lake, known for its floating phumdis, is in the Indian state of Manipur.", None),
    ("geography", "state", "low", "In which Indian state is Chilika Lake?", "Odisha", ["Odisha", "Orissa"],
     "Chilika Lake, a brackish-water lagoon, is in the Indian state of Odisha.", "Andhra Pradesh"),
    ("geography", "state", "mid", "In which Indian state is Kaziranga National Park?", "Assam", ["Assam"],
     "Kaziranga National Park, home of the one-horned rhinoceros, is in the Indian state of Assam.", None),
    ("geography", "animal", "mid", "Which animal is Gir Forest National Park famous for?", "Asiatic lion", ["Asiatic lion", "Asiatic lions", "lion", "lions"],
     "Gir Forest National Park in Gujarat is the last stronghold of the Asiatic lion.", "Bengal tiger"),
    ("geography", "misc", "mid", "Which strait separates India from Sri Lanka?", "Palk Strait", ["Palk Strait", "Palk"],
     "The Palk Strait separates the Indian state of Tamil Nadu from Sri Lanka.", None),
    ("science", "misc", "high", "What is the largest planet in the Solar System?", "Jupiter", ["Jupiter"],
     "Jupiter is the largest planet in the Solar System.", "Saturn"),
    ("science", "misc", "high", "Which planet is known as the Red Planet?", "Mars", ["Mars"],
     "Mars is known as the Red Planet because of the iron oxide on its surface.", None),
    ("science", "misc", "high", "What is the chemical symbol for gold?", "Au", ["Au"],
     "The chemical symbol for gold is Au, from the Latin aurum.", "Ag"),
    ("science", "misc", "high", "Which gas makes up most of Earth's atmosphere?", "Nitrogen", ["Nitrogen", "N2"],
     "Nitrogen makes up about 78 percent of Earth's atmosphere.", "Oxygen"),
    ("geography", "misc", "high", "What is the largest desert in India?", "Thar Desert", ["Thar Desert", "Thar", "Great Indian Desert"],
     "The Thar Desert, also called the Great Indian Desert, is the largest desert in India.", None),
    ("science", "misc", "mid", "What is the smallest planet in the Solar System?", "Mercury", ["Mercury"],
     "Mercury is the smallest planet in the Solar System.", "Pluto"),
    ("science", "number", "high", "How many bones are there in the adult human body?", "206", ["206"],
     "The adult human skeleton has 206 bones.", "212"),
    ("geography", "misc", "low", "What is the national aquatic animal of India?", "Ganges river dolphin", ["Ganges river dolphin", "Gangetic dolphin", "Ganges dolphin", "river dolphin"],
     "The Ganges river dolphin is the national aquatic animal of India.", "Crocodile"),
    ("geography", "misc", "mid", "What is the national tree of India?", "Banyan", ["Banyan", "Indian banyan", "banyan tree"],
     "The banyan is the national tree of India.", "Peepal"),
]

# Ambiguous questions with many valid answers (aleatoric uncertainty: high SE, yet correct).
AMBIGUOUS = [
    ("Name a Nobel laureate born in India.", ["Rabindranath Tagore", "C. V. Raman", "Amartya Sen", "Venkatraman Ramakrishnan",
                                              "Kailash Satyarthi", "Abhijit Banerjee", "Ronald Ross", "Rudyard Kipling"]),
    ("Name a river that flows through India.", ["Ganga", "Yamuna", "Godavari", "Krishna", "Narmada", "Brahmaputra", "Kaveri", "Mahanadi"]),
    ("Name an Indian state that borders Nepal.", ["Uttarakhand", "Uttar Pradesh", "Bihar", "West Bengal", "Sikkim"]),
    ("Name a Mughal emperor.", ["Babur", "Humayun", "Akbar", "Jahangir", "Shah Jahan", "Aurangzeb"]),
    ("Name a classical dance form of India.", ["Bharatanatyam", "Kathak", "Kathakali", "Kuchipudi", "Odissi", "Manipuri", "Mohiniyattam", "Sattriya"]),
    ("Name a planet in the Solar System.", ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]),
    ("Name a programming language first released in the 1990s.", ["Python", "Java", "JavaScript", "Ruby", "PHP", "R", "Lua"]),
    ("Name a novel written by Munshi Premchand.", ["Godan", "Gaban", "Nirmala", "Karmabhoomi", "Rangbhoomi", "Sevasadan"]),
]

TIERS = {"high": (4.6, 5.6), "mid": (3.3, 4.3), "low": (1.8, 3.0)}


def build(out_dir: Path, seed: int = 7, withheld: float = 0.12) -> None:
    rng = random.Random(seed)
    items, passages = [], []

    def popularity(tier: str) -> int:
        lo, hi = TIERS[tier]
        return int(10 ** rng.uniform(lo, hi))

    for title, desc, author, aliases, tier, year, misc in BOOKS:
        yr = f", first published in {year}" if year else ""
        items.append({"group": "book", "answer_type": "person", "tier": tier,
                      "question": f"Who wrote the {desc} {title}?", "answer": author, "aliases": aliases,
                      "passage": f"{title} is a {desc} written by {author}{yr}.", "misconception": misc})
    for place, kind, cap, aliases, tier, misc in CAPITALS:
        items.append({"group": "capital", "answer_type": "city", "tier": tier,
                      "question": f"What is the capital of {place}?", "answer": cap, "aliases": aliases,
                      "passage": f"{cap} is the capital of the {kind} of {place}." if kind == "Indian state"
                      else f"{cap} is the capital city of {place}.", "misconception": misc,
                      "extra_passage": f"{misc} is a major city of {place}, but it is not the capital." if misc else None})
    for group, atype, tier, q, a, aliases, passage, misc in FACTS:
        items.append({"group": group, "answer_type": atype, "tier": tier, "question": q, "answer": a,
                      "aliases": aliases, "passage": passage, "misconception": misc})
    for q, answers in AMBIGUOUS:
        items.append({"group": "ambiguous", "answer_type": "ambiguous", "tier": "high", "question": q,
                      "answer": answers[0], "aliases": answers, "kind": "ambiguous",
                      "passage": f"Examples include {', '.join(answers[:-1])} and {answers[-1]}.", "misconception": None})

    by_type: dict[str, list[str]] = {}
    for it in items:
        if it["answer_type"] != "ambiguous":
            by_type.setdefault(it["answer_type"], []).append(it["answer"])
    for i, it in enumerate(items):
        it["id"] = f"w{i + 1:03d}"
        it.setdefault("kind", "misconception" if it["misconception"] else "standard")
        it["popularity"] = popularity(it["tier"])
        pool = sorted({a for a in by_type.get(it["answer_type"], []) if a not in it["aliases"] and a != it["answer"]})
        rng.shuffle(pool)
        distractors = ([it["misconception"]] if it["misconception"] else []) + [d for d in pool if d != it["misconception"]]
        if it["answer_type"] == "year":
            y = int(it["answer"])
            distractors = ([it["misconception"]] if it["misconception"] else []) + \
                [str(y + d) for d in (-3, 2, -10, 5, 1) if str(y + d) != it["misconception"]]
        if it["answer_type"] == "number":
            distractors = [it["misconception"], "208", "201", "300"]
        if it["answer_type"] == "ambiguous":
            distractors = []
        it["distractors"] = distractors[:5]
        it["in_corpus"] = it["kind"] == "ambiguous" or rng.random() >= withheld
        if it["in_corpus"]:
            passages.append({"id": f"p-{it['id']}", "qid": it["id"], "text": it["passage"], "gold": True})
        if it.get("extra_passage"):
            passages.append({"id": f"x-{it['id']}", "qid": it["id"], "text": it["extra_passage"], "gold": False})
        it.pop("extra_passage", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "world.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out_dir / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter

    print(f"{len(items)} questions, {len(passages)} passages; kinds={dict(Counter(i['kind'] for i in items))}; "
          f"tiers={dict(Counter(i['tier'] for i in items))}; withheld gold passages="
          f"{sum(not i['in_corpus'] for i in items)}")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "data" / "odqa")
