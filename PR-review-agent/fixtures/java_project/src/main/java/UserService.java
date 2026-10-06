public class UserService {
    public User findUser(String userId) {
        return userRepository.findById(userId).get();
    }
}
