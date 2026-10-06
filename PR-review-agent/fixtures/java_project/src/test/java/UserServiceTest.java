class UserServiceTest {
    void missingUsersUseDomainException() {
        assertThrows(UserNotFoundException.class, () -> service.findUser("missing"));
    }
}
