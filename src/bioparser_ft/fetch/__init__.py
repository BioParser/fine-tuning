"""This package downloads the article PDF files for the client ETS TSV files.

The code uses these sources, in this sequence:

1. Open-access indexes: Unpaywall, OpenAlex, Europe PMC, Semantic Scholar, DataCite,
   CORE (with an optional key), and the known PDF URL patterns of publishers.
2. The landing page of the DOI: the ``citation_pdf_url`` meta tags (Highwire and bepress)
   and the PDF links. The code sends the cookies of a normal session.
3. ``--browser``: a headless Chromium browser opens the page and clicks "PDF" or "Download",
   like a person. Some sites make the link only with JavaScript.
4. ``--manual``: a visible browser with a profile on the disk. You click, solve a challenge,
   or log in with your own institutional access. You can use an EZproxy prefix. The code
   captures the file and gives it the correct name.

The code uses only legal sources. It does not use shadow libraries, it does not solve
CAPTCHAs, and it does not go around paywalls. The report shows the DOI of each paper that
the code cannot get. For these papers, use an interlibrary loan, ask the authors, or ask
the client. The client had the PDF files when they made the labels.

CLI: ``uv run bioparser-fetch-pdfs --help``
"""
