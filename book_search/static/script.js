/**
 * 도서 상세 정보 팝업을 표시합니다.
 * index.html에 정의된 템플릿에 책 정보를 채워 넣습니다.
 * @param {object} book - 표시할 책의 정보 객체
 */
function showBookDetails(book) {
    console.log('상세 정보 팝업 표시:', book);
    const modal = document.getElementById('bookDetailsModal');

    // 이미지 정보 채우기
    const imageEl = document.getElementById('details-image');
    const imagePlaceholderEl = document.getElementById('details-image-placeholder');
    if (book.image && book.image.trim() !== '') {
        imageEl.src = book.image;
        imageEl.alt = book.title;
        imageEl.style.display = 'block';
        imagePlaceholderEl.style.display = 'none';
    } else {
        imageEl.style.display = 'none';
        imagePlaceholderEl.style.display = 'flex';
    }

    // 기본 정보 채우기 (제목, 저자, 링크)
    document.getElementById('details-title').textContent = book.title;
    document.getElementById('details-author').textContent = book.author;
    document.getElementById('details-reading-link').href = `https://read365.edunet.net/PureScreen/SchoolSearchResult?searchKeyword=${encodeURIComponent(book.title)}&provCode=I10&neisCode=I100000144&schoolName=종촌고등학교`;

    // 헬퍼 함수: 정보 아이템의 값을 채우고, 값이 없으면 숨김
    const populateInfoItem = (elementId, value) => {
        const container = document.getElementById(`${elementId}-container`);
        if (value && String(value).trim() !== '') {
            document.getElementById(elementId).textContent = value;
            container.classList.remove('hidden');
        } else {
            container.classList.add('hidden');
        }
    };

    // 상세 정보 그리드 채우기
    populateInfoItem('details-publisher', book.publisher);
    populateInfoItem('details-year', book.year);
    populateInfoItem('details-category', book.category);
    populateInfoItem('details-location', book.location);
    populateInfoItem('details-isbn', book.isbn);
    populateInfoItem('details-pages', book.pages);

    // 내용 소개 섹션 채우기
    const descriptionContainer = document.getElementById('details-description-container');
    if (book.description && book.description.trim() !== '' && book.description !== '설명이 없습니다.') {
        document.getElementById('details-description').textContent = book.description;
        descriptionContainer.classList.remove('hidden');
    } else {
        descriptionContainer.classList.add('hidden');
    }

    modal.style.display = 'flex';
}

let isFirstSearch = true;
const API_BASE_URL = window.location.origin;

async function searchBooks(query, topK = 6) {
    try {
        console.log('🔍 검색 요청:', { query, topK });
        
        const requestBody = {
            npz: "books_emb.npz",
            meta: "books_meta.csv",
            query: query,
            top_k: topK,
            source_csv: "book.csv",
            source_id_col: "No",
            randomize: true,
            tau: 0.01
        };
        
        console.log('📡 API 요청:', requestBody);
        
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        console.log('📥 응답 상태:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            console.error('서버 오류:', errorText);
            throw new Error(`서버 오류: ${response.status}`);
        }

        const data = await response.json();
        console.log('📥 API 응답:', data);

        let results = [];
        if (data && Array.isArray(data.results)) {
            results = data.results;
        } else {
            throw new Error('예상치 못한 응답 형식');
        }
           
        // API 응답의 다양한 필드명을 일관된 이름으로 매핑
        const normalizedResults = results.map((book, index) => {
            return {
                id: book.id || book.No || book.no || book.번호 || `book_${index}`,
                title: book.자료명 || '',
                author: book.저자 || '',
                publisher: book.publisher || book.출판사 || book.발행자 || book.Publisher || book.출판 || '',
                year: book.year || book.출판년도 || book.발행년도 || book.Year || book.출판연도 || book.년도 || '',
                isbn: book.isbn || book.ISBN || book.Isbn || '',
                category: book.category || book.분류 || book.Category || book.장르 || book.카테고리 || '',
                location: book.location || book.위치 || book.소장위치 || book.Location || book.서가 || '',
                status: book.status || book.상태 || 'available',
                description: book.줄거리 || '설명이 없습니다.',
                pages: book.pages || book.페이지 || book.Pages || book.쪽수 || '',
                image: book.이미지 || '',
                rawData: book
            };
        }).filter(book => book.title && book.title.trim() !== '');
        
        console.log('정규화된 결과:', normalizedResults);

        if (normalizedResults.length === 0) {
            throw new Error('유효한 도서 데이터가 없습니다');
        }

        return { results: normalizedResults };

    } catch (error) {
        console.error('도서 검색 오류:', error);
        addChatMessage('ai', `⚠️ 검색 중 오류가 발생했습니다.\n\n서버가 실행 중인지 확인해주세요.\n오류 내용: ${error.message}`);
        return { results: [] };
    }
}

async function performSearch(event) {
    if (event) event.preventDefault();
    const searchTerm = document.getElementById('searchInput').value.trim();

    if (searchTerm) {
        document.getElementById('searchInput').value = '';
        resetTextareaHeight();

        if (isFirstSearch) {
            moveSearchBarToTop();
            isFirstSearch = false;
        }
        
        addChatMessage('user', searchTerm);
        const loadingDiv = addChatMessage('ai', `"${searchTerm}"와 관련된 도서를 검색하고 있습니다...`);

        try {
            const searchResults = await searchBooks(searchTerm);
            if (loadingDiv?.parentNode) {
                loadingDiv.remove();
            }

            const response = `"${searchTerm}"와 관련된 도서를 찾아드렸습니다. 다음 추천 도서들을 확인해보세요:`;
            addChatMessage('ai', response);

            if (searchResults?.results?.length > 0) {
                addBookGallery(searchResults.results);
            } else {
                addChatMessage('ai', '죄송합니다. 관련된 도서를 찾을 수 없습니다. 다른 키워드로 검색해보세요.');
            }
        } catch (error) {
            console.error('검색 오류:', error);
            if (loadingDiv?.parentNode) {
                loadingDiv.remove();
            }
            addChatMessage('ai', '도서 검색 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
        }
    }
}

/**
 * 첫 검색 시, 검색창을 상단으로 이동시키는 UI 변경을 처리합니다.
 * body에 'search-active' 클래스를 추가하여 CSS가 레이아웃을 변경하도록 합니다.
 */
function moveSearchBarToTop() {
    const searchContainer = document.querySelector('.search-container');
    const chatContainer = document.getElementById('chat-container');
    const searchInput = document.getElementById('searchInput');

    document.body.classList.add('search-active');
    searchInput.placeholder = '무엇이든 물어보세요';
    
    // 클래스 추가로 인한 스타일 변경이 렌더링된 후 높이를 계산하기 위해 setTimeout 사용
    setTimeout(() => {
        const headerHeight = Math.ceil(searchContainer.getBoundingClientRect().height);
        chatContainer.style.top = headerHeight + 'px';
        chatContainer.style.height = `calc(100vh - ${headerHeight}px)`;
    }, 0);
}

function addChatMessage(type, message) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}-message`;
    messageDiv.textContent = message;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv;
}

function addBookGallery(books) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chat-message ai-message gallery-container';
    const galleryDiv = document.createElement('div');
    galleryDiv.className = 'book-gallery';
    const top6Books = books.slice(0, 6);

    top6Books.forEach(book => {
        const bookItem = document.createElement('div');
        bookItem.className = 'book-item';
        bookItem.onclick = () => showBookDetails(book);
        
        let bookCoverHTML = (book.image && book.image.trim() !== '')
            ? `<img src="${book.image}" alt="${book.title}" class="book-cover-img" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
               <div class="book-cover" style="display:none;">📚</div>`
            : `<div class="book-cover">📚</div>`;
        
        const truncatedTitle = book.title.length > 40 ? book.title.substring(0, 40) + '...' : book.title;
        const truncatedAuthor = book.author.length > 20 ? book.author.substring(0, 20) + '...' : book.author;
        
        bookItem.innerHTML = `
            ${bookCoverHTML}
            <div class="book-title" title="${book.title}">${truncatedTitle}</div>
            <div class="book-author" title="${book.author}">${truncatedAuthor}</div>
            <div class="book-status ${book.status}">${book.status === 'available' ? '대출가능' : '대출중'}</div>
        `;
        galleryDiv.appendChild(bookItem);
    });

    messageDiv.appendChild(galleryDiv);
    if (books.length > 6) {
        const moreButton = document.createElement('button');
        moreButton.className = 'more-books-btn';
        moreButton.textContent = `더 많은 추천 보기 (${books.length - 6}권 더)`;
        moreButton.onclick = () => showAllBooks(books);
        messageDiv.appendChild(moreButton);
    }
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showAllBooks(books) {
    const modal = document.getElementById('bookModal');
    document.getElementById('modalTitle').textContent = `전체 추천 도서 목록 (${books.length}권)`;
    const bookList = document.getElementById('bookList');
    bookList.innerHTML = '';

    books.forEach(book => {
        const bookCard = document.createElement('div');
        bookCard.className = 'book-card';
        bookCard.onclick = () => { closeModal(); showBookDetails(book); };
        
        let bookCoverHTML = (book.image && book.image.trim() !== '')
            ? `<img src="${book.image}" alt="${book.title}" class="book-cover-img" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
               <div class="book-cover" style="display:none;">📚</div>`
            : `<div class="book-cover">📚</div>`;
        
        const truncatedDescription = book.description.length > 100 ? book.description.substring(0, 100) + '...' : book.description;
        
        bookCard.innerHTML = `
            ${bookCoverHTML}
            <div class="book-title" title="${book.title}">${book.title}</div>
            <div class="book-author" title="${book.author}">${book.author}</div>
            ${book.publisher ? `<div class="book-publisher">출판사: ${book.publisher}</div>` : ''}
            ${book.year ? `<div class="book-year">출판연도: ${book.year}</div>` : ''}
            ${book.category ? `<div class="book-category">분류: ${book.category}</div>` : ''}
            ${book.location ? `<div class="book-location">위치: ${book.location}</div>` : ''}
            <div class="book-description" title="${book.description}">${truncatedDescription}</div>
            <div class="book-status ${book.status}">${book.status === 'available' ? '대출가능' : '대출중'}</div>
        `;
        bookList.appendChild(bookCard);
    });
    modal.style.display = 'flex';
}

function closeModal() {
    document.getElementById('bookModal').style.display = 'none';
}

function closeDetailsModal() {
    document.getElementById('bookDetailsModal').style.display = 'none';
}

function adjustTextareaHeight(textarea) {
    textarea.style.height = '50px';
    const scrollHeight = textarea.scrollHeight;
    const maxHeight = 200;
    if (scrollHeight > 50) {
        textarea.style.height = Math.min(scrollHeight, maxHeight) + 'px';
    }
    textarea.style.overflowY = scrollHeight > maxHeight ? 'auto' : 'hidden';
}

function resetTextareaHeight() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.style.height = '50px';
        searchInput.style.overflowY = 'hidden';
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const searchForm = document.getElementById('searchForm');
    const searchInput = document.getElementById('searchInput');
    
    searchForm?.addEventListener('submit', performSearch);
    
    if (searchInput) {
        searchInput.addEventListener('input', () => adjustTextareaHeight(searchInput));
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                searchForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
            }
        });
        searchInput.addEventListener('focus', () => {
            if (searchInput.value.trim() === '') searchInput.style.height = '50px';
        });
    }
});

window.onclick = function(event) {
    const listModal = document.getElementById('bookModal');
    const detailsModal = document.getElementById('bookDetailsModal');
    
    if (event.target === listModal) closeModal();
    if (event.target === detailsModal) closeDetailsModal();
}